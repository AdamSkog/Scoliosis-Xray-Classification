import copy
import datetime
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs=25,
    model_log_dir="logs/runs",
):
    """
    Trains a model using the specified criterion, optimizer, and data loaders.
    Args:
        model (nn.Module): The model to be trained.
        criterion (nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        dataloaders (dict): A dictionary containing the data loaders for the training and validation sets.
        dataset_sizes (dict): A dictionary containing the sizes of the training and validation datasets.
        device (torch.device): The device on which the model and data will be loaded.
        num_epochs (int, optional): The number of epochs to train the model (default: 25).
        model_log_dir (str, optional): The directory to save the TensorBoard logs (default: "logs/runs").
    Returns:
        nn.Module: The trained model.
    """
    run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(model_log_dir, run_name)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # Log the model graph
    sample_inputs, _ = next(iter(dataloaders["train"]))
    writer.add_graph(model, sample_inputs.to(device))

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            dataloader = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize()} Epoch {epoch}/{num_epochs}",
            )

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                dataloader.set_postfix({"Loss": loss.item()})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
            )

            # Log metrics to TensorBoard
            writer.add_scalar(f"{phase.capitalize()} Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase.capitalize()} Accuracy", epoch_acc, epoch)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    logger.info(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_model_wts)

    # Close the TensorBoard writer
    writer.close()

    return model
