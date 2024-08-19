import logging

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def test_model(model, dataloaders, dataset_sizes, device):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        dataloader = tqdm(dataloaders["test"], desc="Testing")

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes["test"]
    logger.info(f"Test Acc: {test_acc:.4f}")
    return test_acc
