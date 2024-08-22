import logging
import os

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Get data loaders for training, validation, and testing datasets.

    Args:
        data_dir (str): Directory path containing the dataset.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.

    Returns:
        tuple: A tuple containing the data loaders, dataset sizes, and class names.
    """
    
    # Define data transformations
    # Data normalization values determined from resnet50 model
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Load the full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms["train"])

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    logger.info("Dataset split into train, validation, and test sets.")
    logger.info("Applying data transformations...")

    # Apply data transformations to train, validation, and test sets
    train_dataset.dataset.transform = data_transforms["train"]
    val_dataset.dataset.transform = data_transforms["val"]
    test_dataset.dataset.transform = data_transforms["test"]

    logger.info("Data transformations applied successfully.")
    logger.info("Creating data loaders...")

    # Create data loaders for train, validation, and test sets
    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "test": DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
    }

    # Calculate dataset sizes
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }

    # Get class names
    class_names = full_dataset.classes

    logger.info(f"Class names: {class_names}")
    logger.info("Data loaders created successfully.")

    return dataloaders, dataset_sizes, class_names
