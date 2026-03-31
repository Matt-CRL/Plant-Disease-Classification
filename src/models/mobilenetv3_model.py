from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


def build_mobilenetv3_small(num_classes: int, freeze_features: bool = False):
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    return model