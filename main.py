import io
import os
import base64
import logging
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms as T

from src.model import load_mobilenet_model

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="Detects DR severity from retinal fundus images using MobileNetV3.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "artifacts/mobilenet.pt"
IMAGE_SIZE = 224

DR_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

DR_ADVICE = {
    0: "No signs of diabetic retinopathy. Maintain regular annual eye screenings and keep blood sugar well-controlled.",
    1: "Mild DR detected. Consult your ophthalmologist. Tighten blood sugar and blood pressure control.",
    2: "Moderate DR detected. Prompt ophthalmology referral recommended. Monitor every 6 months.",
    3: "Severe DR detected. Urgent ophthalmology referral required. High risk of progression.",
    4: "Proliferative DR detected. URGENT: Immediate specialist care required. High risk of severe vision loss.",
}

DR_SEVERITY_COLOR = {
    0: "green",
    1: "yellow",
    2: "orange",
    3: "red",
    4: "darkred",
}

# ── Preprocessing (ImageNet standard — matches MobileNet training) ─────────────
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Load Model ────────────────────────────────────────────────────────────────
model = None

def download_model_if_needed():
    """Download model from Google Drive if not present locally."""
    if os.path.exists(CHECKPOINT_PATH):
        logger.info("✅ Model file already exists.")
        return
    
    FILE_ID = "1v9qt0UFsIj7PgkDAjzetVR7Z9mYeVLTy"  # ← paste your Google Drive file ID
    
    logger.info("⬇️  Downloading model from Google Drive...")
    os.makedirs("artifacts", exist_ok=True)
    
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    
    import urllib.request
    try:
        urllib.request.urlretrieve(url, CHECKPOINT_PATH)
        logger.info("✅ Model downloaded successfully!")
    except Exception as e:
        logger.error("❌ Download failed: %s", str(e))


def load_model():
    global model
    download_model_if_needed()
    try:
        model = load_mobilenet_model(CHECKPOINT_PATH, num_classes=5)
        logger.info("✅ MobileNetV3 model loaded from %s", CHECKPOINT_PATH)
    except FileNotFoundError:
        logger.warning("⚠️  Model not found. Running in DEMO mode.")
        model = None
    except RuntimeError as e:
        logger.error("❌ Model load failed: %s", str(e))
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()

# ── Response Schema ───────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    predicted_class: int
    label: str
    confidence: float
    all_confidences: dict
    medical_advice: str
    severity_color: str
    gradcam_image: Optional[str] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    return img, tensor


def generate_gradcam(model_obj, tensor: torch.Tensor, target_class: int) -> Optional[str]:
    try:
        import cv2

        gradients = []
        activations = []

        # Find last conv layer
        target_layer = None
        for module in model_obj.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            return None

        def fwd_hook(module, input, output):
            activations.append(output.detach())

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        fh = target_layer.register_forward_hook(fwd_hook)
        bh = target_layer.register_full_backward_hook(bwd_hook)

        # Need grad — clone with grad enabled
        inp = tensor.clone().requires_grad_(True)
        output = model_obj(inp)
        model_obj.zero_grad()
        output[0, target_class].backward()

        fh.remove()
        bh.remove()

        if not gradients or not activations:
            return None

        grad = gradients[0].squeeze(0)
        act  = activations[0].squeeze(0)
        weights = grad.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
        cam = cam.numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        # Reconstruct original image
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        orig = tensor.squeeze(0).permute(1, 2, 0).numpy()
        orig = np.clip((orig * std + mean) * 255, 0, 255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)
        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf).decode("utf-8")

    except Exception as e:
        logger.warning("Grad-CAM failed: %s", e)
        return None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "DR Detection API (MobileNetV3)",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/health", "/labels"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/labels")
def get_labels():
    return {"labels": DR_LABELS, "advice": DR_ADVICE}


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Only JPEG/PNG images supported.")

    try:
        image_bytes = await file.read()
        pil_img, tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if model is not None:
        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
        gradcam_b64 = generate_gradcam(model, tensor, int(probs.argmax()))
    else:
        # Demo mode — random predictions
        probs = torch.softmax(torch.randn(5), dim=0)
        gradcam_b64 = None

    predicted_idx  = int(probs.argmax())
    confidence     = float(probs[predicted_idx])
    all_confidences = {DR_LABELS[i]: round(float(probs[i]), 4) for i in range(5)}

    return PredictionResult(
        predicted_class=predicted_idx,
        label=DR_LABELS[predicted_idx],
        confidence=round(confidence * 100, 2),
        all_confidences=all_confidences,
        medical_advice=DR_ADVICE[predicted_idx],
        severity_color=DR_SEVERITY_COLOR[predicted_idx],
        gradcam_image=gradcam_b64,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)