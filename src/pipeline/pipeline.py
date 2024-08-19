import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.components import *
from src.utils.common import *

logger = logging.getLogger(__name__)


def pipeline():
    """
    Runs the pipeline for training and testing a model for Scoliosis X-ray classification.
    Returns:
        None
    """
    params = read_yaml(Path("params.yaml"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"**Using device: {device}**")

    dataloaders, dataset_sizes, class_names = get_dataloaders(params["data_dir"])

    model = initialize_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model,
        criterion,
        optimizer,
        dataloaders,
        dataset_sizes,
        device,
        num_epochs=params["num_epochs"],
        model_log_dir=params["model_log_dir"],
    )

    # Save the trained model
    model_save_path = os.path.join(params["model_save_dir"], "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Test the model with the test dataset
    logger.info("Testing the model")
    test_acc = test_model(model, dataloaders, dataset_sizes, device)
