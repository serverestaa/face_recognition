import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Constants
BATCH_SIZE = 32
NUM_PAIRS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset Definitions
class FaceDataset(Dataset):
    """
    Dataset for handling face images and their corresponding labels.

    Args:
        root_dir (str): Root directory containing image folders.
        transform (callable, optional): Transformations to apply to images. Default is None.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels, self.label_mapping = self._load_data()

    def _load_data(self):
        image_paths = []
        labels = []
        label_mapping = {}
        current_label = 0

        for folder_name in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                label_mapping[folder_name] = current_label
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                        image_paths.append(os.path.join(folder_path, filename))
                        labels.append(current_label)
                current_label += 1
        return image_paths, labels, label_mapping

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class FaceVerificationDataset(Dataset):
    """
    Dataset for generating pairs of images for verification tasks.

    Args:
        dataset (FaceDataset): Instance of FaceDataset containing images and labels.
        num_pairs (int): Number of pairs to generate.
    """

    def __init__(self, dataset: FaceDataset, num_pairs: int = NUM_PAIRS):
        self.dataset = dataset
        self.num_pairs = num_pairs
        self.pairs = []
        self.labels = []
        self._generate_pairs()

    def _generate_pairs(self):
        class_indices = {}
        for idx, label in enumerate(self.dataset.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        positive_pairs = 0
        labels_list = list(class_indices.keys())
        while positive_pairs < self.num_pairs // 2:
            label = random.choice(labels_list)
            if len(class_indices[label]) >= 2:
                idx1, idx2 = random.sample(class_indices[label], 2)
                self.pairs.append((idx1, idx2))
                self.labels.append(1)
                positive_pairs += 1

        negative_pairs = 0
        while negative_pairs < self.num_pairs // 2:
            label1, label2 = random.sample(labels_list, 2)
            if label1 != label2:
                idx1 = random.choice(class_indices[label1])
                idx2 = random.choice(class_indices[label2])
                self.pairs.append((idx1, idx2))
                self.labels.append(0)
                negative_pairs += 1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]
        label = self.labels[idx]
        return img1, img2, label


# Model Evaluation
def evaluate_model(model, dataset, device):
    """
    Evaluates the model using a verification dataset.

    Args:
        model (torch.nn.Module): Trained face recognition model.
        dataset (FaceDataset): Dataset containing face images and labels.
        device (torch.device): Device for computations (CPU or GPU).

    Returns:
        tuple: Best threshold, accuracy, ROC AUC, distances, and labels.
    """
    verification_dataset = FaceVerificationDataset(dataset)
    verification_loader = DataLoader(verification_dataset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings1, embeddings2, labels = [], [], []
    pairs, scores = [], []
    model.eval()

    with torch.no_grad():
        for img1, img2, label in tqdm(verification_loader, desc="Evaluating"):
            img1, img2 = img1.to(device), img2.to(device)
            emb1, emb2 = model(img1), model(img2)
            embeddings1.append(emb1.cpu().numpy())
            embeddings2.append(emb2.cpu().numpy())
            labels.extend(label.numpy())
            for idx in range(len(label)):
                pairs.append(verification_dataset.pairs[len(pairs)])

    embeddings1 = np.vstack(embeddings1)
    embeddings2 = np.vstack(embeddings2)
    labels = np.array(labels)
    distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)

    # Compute metrics
    similarity_scores = -distances
    fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
    roc_auc = auc(fpr, tpr)

    accuracies = [accuracy_score(labels, similarity_scores >= threshold) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(accuracies)]
    best_accuracy = np.max(accuracies)

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
    print(f"AUC: {roc_auc:.4f}")

    return best_threshold, best_accuracy, roc_auc, distances, labels


# Visualization
def plot_tsne(embeddings, labels):
    """
    Plots t-SNE visualization of embeddings.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (np.ndarray): Corresponding labels.
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
        hue=labels, palette="bright", legend=None
    )
    plt.title("t-SNE Visualization of Embeddings")
    plt.show()

def plot_distance_distribution(distances, labels):
    """
    Plots distance distribution for positive and negative distances.

    Args:
        distances (np.ndarray): Distances.
        labels (np.ndarray): Corresponding labels.
    """
    # Split distances into positive and negative based on labels
    pos_distances = [dist for dist, label in zip(distances, labels) if label == 1]
    neg_distances = [dist for dist, label in zip(distances, labels) if label == 0]
    
    # Plotting the distributions
    plt.figure(figsize=(10, 6))
    plt.hist(pos_distances, bins=30, alpha=0.6, label='Positive Pair Distances')
    plt.hist(neg_distances, bins=30, alpha=0.6, label='Negative Pair Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Positive and Negative Pair Distances')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ROOT_DIR = "updated_dataset"
    model_path = "model_scripted_EfficientNetv3_10epochs.pt"

    # Load model
    model = torch.jit.load(model_path).to(DEVICE)
    model.eval()

    # Define dataset and transformations
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = FaceDataset(ROOT_DIR, transform=val_transform)

    # Evaluate model
    best_threshold, best_accuracy, roc_auc, distances, labels = evaluate_model(model, dataset, DEVICE)
    print(f"Best Threshold: {best_threshold:.4f}, Best Accuracy: {best_accuracy * 100:.2f}%")

    #  Plot the distance distribution
    plot_distance_distribution(distances, labels)
    # Plot results
    embeddings = np.vstack([model(dataset[i][0].unsqueeze(0).to(DEVICE)).detach().cpu().numpy() for i in range(len(dataset))])
    plot_tsne(embeddings, dataset.labels)