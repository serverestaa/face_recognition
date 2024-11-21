import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """
    Dataset for handling face images and their corresponding labels.

    Args:
        root_dir (str): Root directory containing image folders.
        selected_folders (list): List of folder names to include in the dataset.
        label_mapping (dict): Mapping of folder names to numerical labels.
        transform (callable, optional): Transformations to apply to images. Default is None.
    """

    def __init__(self, root_dir: str, selected_folders: list, label_mapping: dict, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = label_mapping
        self.image_paths, self.labels = self._load_data(selected_folders)

    def _load_data(self, selected_folders: list):
        """
        Loads image paths and labels from the selected folders.

        Args:
            selected_folders (list): List of folder names to process.

        Returns:
            tuple: List of image paths and their corresponding labels.
        """
        image_paths = []
        labels = []

        for folder in selected_folders:
            folder_path = os.path.join(self.root_dir, folder)
            label = self.label_mapping[folder]
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                    image_paths.append(os.path.join(folder_path, filename))
                    labels.append(label)

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TripletDataset(Dataset):
    """
    Dataset for generating triplets (anchor, positive, negative) for training with triplet loss.

    Args:
        dataset (FaceDataset): Instance of FaceDataset containing images and labels.
    """

    def __init__(self, dataset: FaceDataset):
        self.dataset = dataset
        self.class_indices = {}
        self.valid_indices = []

        # Organize indices by class label
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        # Keep only classes with at least 2 samples
        self.class_indices = {k: v for k, v in self.class_indices.items() if len(v) >= 2}

        # Generate a list of valid indices
        self.valid_indices = [idx for indices in self.class_indices.values() for idx in indices]

        if not self.valid_indices:
            raise ValueError("No valid classes with at least 2 samples each were found in the dataset.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Select an anchor
        anchor_idx = self.valid_indices[idx]
        anchor_label = self.dataset.labels[anchor_idx]

        # Select a positive example
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
            positive_idx = random.choice(self.class_indices[anchor_label])

        anchor_img = self.dataset[anchor_idx][0]
        positive_img = self.dataset[positive_idx][0]

        # Select a negative example
        negative_label = random.choice(
            [label for label in self.class_indices.keys() if label != anchor_label]
        )
        negative_idx = random.choice(self.class_indices[negative_label])
        negative_img = self.dataset[negative_idx][0]

        return anchor_img, positive_img, negative_img


class TripletLoss(nn.Module):
    """
    Triplet Loss for training deep metric learning models.

    Args:
        alpha (float): Margin value to separate positive and negative pairs. Default is 1.0.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def calc_euclidean(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Euclidean distance between two tensors.

        Args:
            x1 (torch.Tensor): First tensor.
            x2 (torch.Tensor): Second tensor.

        Returns:
            torch.Tensor: Element-wise Euclidean distance.
        """
        return torch.sqrt((x1 - x2).pow(2).sum(1))

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the triplet loss.

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            positive (torch.Tensor): Positive embeddings.
            negative (torch.Tensor): Negative embeddings.

        Returns:
            torch.Tensor: Mean triplet loss.
        """
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.alpha)
        return losses.mean()


def unnormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Unnormalizes an image tensor.

    Args:
        img_tensor (torch.Tensor): Normalized image tensor.

    Returns:
        np.ndarray: Unnormalized image array in [0, 1] range.
    """
    img = img_tensor.clone().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img
