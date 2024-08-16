import torch.nn as nn
from torchvision import models


def initialize_model(device):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification
    model = model.to(device)
    return model
