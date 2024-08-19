import torch.nn as nn
from torchvision import models


def initialize_model(device):
    """
    Initializes a ResNet-50 model for binary classification.
    Args:
        device (torch.device): The device to use for model computation.
    Returns:
        torch.nn.Module: The initialized ResNet-50 model.
    """
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification
    model = model.to(device)
    return model
