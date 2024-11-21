import torch
from tqdm import tqdm


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to use for computations (CPU or GPU).

    Returns:
        float: Average loss for the training epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (anchors, positives, negatives) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="Training",
    ):
        # Move data to device
        anchors, positives, negatives = (
            anchors.to(device),
            positives.to(device),
            negatives.to(device),
        )

        # Forward pass
        optimizer.zero_grad()
        anchor_embeddings = model(anchors)
        positive_embeddings = model(positives)
        negative_embeddings = model(negatives)

        # Compute loss
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Train Epoch: \tLoss: {avg_loss:.4f}")
    return avg_loss


def validate_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Validates the model for one epoch.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to use for computations (CPU or GPU).

    Returns:
        float: Average loss for the validation epoch.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (anchors, positives, negatives) in tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="Validation",
        ):
            # Move data to device
            anchors, positives, negatives = (
                anchors.to(device),
                positives.to(device),
                negatives.to(device),
            )

            # Forward pass
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)

            # Compute loss
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

            # Accumulate loss
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Epoch: \tLoss: {avg_loss:.4f}")
    return avg_loss
