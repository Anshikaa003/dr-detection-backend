import torch
import torch.nn as nn
from torchvision import models


def load_mobilenet_model(checkpoint_path: str, num_classes: int = 5):
    """
    Loads MobileNetV3-Large with custom classifier head.
    Expects model saved as state_dict via torch.save(model.state_dict(), path)

    Actual classifier.3 architecture (read from checkpoint):
        0: Linear(1280, 512)
        1: BatchNorm1d(512)
        2: Hardswish
        3: Dropout
        4: Linear(512, num_classes)
    """
    model = models.mobilenet_v3_large(weights=None)

    # classifier.0 is already Linear(960, 1280) — leave it untouched.
    # Replace only classifier.3 to match the saved weights exactly.
    model.classifier[3] = nn.Sequential(
        nn.Linear(1280, 512),        # classifier.3.0
        nn.BatchNorm1d(512),         # classifier.3.1
        nn.Hardswish(),              # classifier.3.2
        nn.Dropout(p=0.2),           # classifier.3.3
        nn.Linear(512, num_classes), # classifier.3.4
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model