"""
This script will split the data into train and valid sets by using a clustering and picking 1 sample from each cluster.
The clustering will use VGG features and single-linkage clustering calculated via cosine similarity.

In the current folder there should be two directories (images and labels) with corresponding file names (*.jpg and *.txt).
The CLI accepts the ratio between train and valid sets (e.g. -r 30, -r 20) and a seed (--seed 42) for reproducibility.
For experiments, we used -r 20 and -s 42.
"""

import os
import argparse
import numpy as np
import random
import shutil
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split data into train and validation sets using clustering"
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=int,
        default=20,
        help="Percentage of data for validation (e.g. 20 means 80/20 split)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def extract_vgg_features(image_paths, device):
    """Extract features using pretrained VGG16 model."""
    # Load pretrained VGG16 with weights
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    # Remove the final classification layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    # Define preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            weights.transforms(),
        ]
    )

    features = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting features"):
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)  # type: ignore

                # Extract features
                feature = feature_extractor(img_tensor)
                feature = feature.view(feature.size(0), -1).cpu().numpy()
                features.append(feature[0])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Add zeros as features for failed images to maintain order
                features.append(np.zeros(25088))  # 512*7*7 = 25088 for VGG16

    return np.array(features)


def cluster_data(features, n_clusters):
    """Perform single-linkage clustering based on cosine similarity."""
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(features)

    # Convert to distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix

    # Apply single-linkage clustering
    # Correct parameter is 'metric' instead of 'affinity'
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",  # Changed from 'affinity' to 'metric'
        linkage="single",
    )
    return clustering.fit_predict(distance_matrix)


def create_train_val_split(image_paths, cluster_labels):
    """Split data into train and validation sets based on clusters (pick one sample from each cluster)."""
    # Get unique cluster labels
    unique_clusters = np.unique(cluster_labels)

    train_indices = []
    val_indices = []

    # For each cluster, take exactly one sample for validation
    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        # Shuffle indices to randomly select samples
        np.random.shuffle(indices)

        # Take exactly one sample for validation
        val_indices.append(indices[0])

        # Use the rest of the samples for training
        train_indices.extend(indices[1:])

    train_paths = [image_paths[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]

    return train_paths, val_paths


def copy_files(file_paths, src_dir, dst_dir, ext):
    """Copy files from source to destination directory."""
    os.makedirs(dst_dir, exist_ok=True)

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        src_file = os.path.join(src_dir, f"{base_name}.{ext}")
        dst_file = os.path.join(dst_dir, f"{base_name}.{ext}")

        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)


def main():
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device for feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get all image paths
    image_dir = "images"
    label_dir = "labels"

    image_paths = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if not image_paths:
        print(f"No images found in '{image_dir}' directory.")
        return

    print(f"Found {len(image_paths)} images.")

    # Extract VGG features
    features = extract_vgg_features(image_paths, device)

    # Determine number of clusters
    n_valid_images = args.ratio * len(image_paths) // 100
    n_clusters = max(1, n_valid_images)
    print(f"Using {n_clusters} clusters for data splitting.")

    # Perform clustering
    cluster_labels = cluster_data(features, n_clusters)

    # Split into train and validation sets
    train_paths, val_paths = create_train_val_split(image_paths, cluster_labels)

    print(
        f"Split result: {len(train_paths)} training samples, {len(val_paths)} validation samples"
    )

    # Create directories for train and validation sets
    os.makedirs("train/images", exist_ok=True)
    os.makedirs("train/labels", exist_ok=True)
    os.makedirs("valid/images", exist_ok=True)
    os.makedirs("valid/labels", exist_ok=True)

    # Copy images to respective directories
    print("Copying train images...")
    for img_path in tqdm(train_paths):
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join("train/images", img_name))

        # Copy corresponding label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join("train/labels", label_name))

    print("Copying validation images...")
    for img_path in tqdm(val_paths):
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join("valid/images", img_name))

        # Copy corresponding label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join("valid/labels", label_name))

    print(
        f"Data split complete: {len(train_paths)} training samples ({100 - args.ratio}%) and {len(val_paths)} validation samples ({args.ratio}%)"
    )


if __name__ == "__main__":
    main()
