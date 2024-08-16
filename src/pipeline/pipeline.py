import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.components import *


def main():
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = "data"
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)

    model = initialize_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device)

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


if __name__ == "__main__":
    main()
