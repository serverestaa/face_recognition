import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from model_training import train_epoch, validate_epoch
from preprocess import remove_small_folders, visualize_face_extraction, crop_and_save_faces
from utils import FaceDataset, TripletDataset, TripletLoss


# Constants
DIR_PATH = "lfw-deepfunneled"
DEST_DIR = "lfw-deepfunneled_cropped"
MIN_FILES_COUNT = 4
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_data_pipeline():
    """
    Sets up the data pipeline: preprocessing, transformations, dataset creation.

    Returns:
        tuple: Training and validation DataLoaders.
    """
    # Remove small folders
    remove_small_folders(DIR_PATH, MIN_FILES_COUNT)

    # Visualize face extraction for a single image
    sample_image = os.path.join(DIR_PATH, "Adam_Sandler", "Adam_Sandler_0002.jpg")
    visualize_face_extraction(sample_image)

    # Crop faces and save them
    crop_and_save_faces(DIR_PATH, DEST_DIR)

    # Create label mapping
    all_folders = sorted(os.listdir(DEST_DIR))
    label_mapping = {folder: idx for idx, folder in enumerate(all_folders)}

    # Split into train and validation sets
    random.seed(42)
    random.shuffle(all_folders)
    split_index = int(0.75 * len(all_folders))
    train_folders = all_folders[:split_index]
    val_folders = all_folders[split_index:]

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = FaceDataset(DEST_DIR, train_folders, label_mapping, transform=train_transform)
    val_dataset = FaceDataset(DEST_DIR, val_folders, label_mapping, transform=val_transform)

    train_loader = DataLoader(
        dataset=TripletDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        dataset=TripletDataset(val_dataset), batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader


def initialize_model():
    """
    Initializes and configures the model for training.

    Returns:
        torch.nn.Module: The model ready for training.
    """
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)

    # Modify the classifier to output embeddings
    features_dim = model.classifier[1].in_features
    model = nn.Sequential(
        *list(model.children())[:-1],  # Remove the classifier
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),  # Output: (batch_size, features_dim)
        nn.Linear(features_dim, 128),  # Embedding layer
        nn.ReLU()
    )
    return model.to(DEVICE)


def plot_losses(train_losses, val_losses):
    """
    Plots training and validation losses.

    Args:
        train_losses (list): Training losses for each epoch.
        val_losses (list): Validation losses for each epoch.
    """
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


def main():
    """
    Main function to set up the data pipeline, train, validate, and save the model.
    """
    # Data Pipeline
    train_loader, val_loader = setup_data_pipeline()

    # Initialize model, loss, and optimizer
    model = initialize_model()
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Validate before training
    print("Validation epoch before training:")
    validate_epoch(model, val_loader, criterion, DEVICE)

    # Training loop
    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
        print("------------------------")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} training:")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plot losses
    plot_losses(train_losses, val_losses)

    # Save model
    torch.save(model.state_dict(), "model_weights_efficientNetv3_10epochs.pth")
    torch.save(model, "model_efficientNetv3_10epochs.pth")

    # Try to script the model
    try:
        scripted_model = torch.jit.script(model)
        scripted_model.save("model_scripted_efficientNetv3_10epochs.pt")
    except Exception as e:
        print(f"Failed to script the model: {e}")


if __name__ == "__main__":
    main()