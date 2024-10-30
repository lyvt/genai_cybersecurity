import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from model import VAE_CNN
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration settings
cfg = {
    'gpu_usage': True,
    'seed': 42,
    'train_dataset_dir': 'data/train',
    'test_dataset_dir': 'data/test',
    'batch_size': 8,
    'save_dir': 'checkpoints',
}


def calculate_metrics(labels, predictions):
    """Calculate evaluation metrics for model performance."""
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    f1_score = metrics.f1_score(labels, predictions)

    # Generate the confusion matrix
    confusion_matrix = metrics.confusion_matrix(labels, predictions)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.pdf')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def test(cfg):
    """Test the model using the specified configuration."""
    device = torch.device('cuda' if cfg['gpu_usage'] else 'cpu')

    # Define image transformation once
    transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets using the same transformation for training and testing
    train_loader = DataLoader(
        datasets.ImageFolder(cfg['train_dataset_dir'], transform=transform),
        batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        datasets.ImageFolder(cfg['test_dataset_dir'], transform=transform),
        batch_size=cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize the AutoEncoder model
    model = VAE_CNN()

    # Load the saved model checkpoint
    model_path = os.path.join(cfg['save_dir'], cfg.get('model_checkpoint', 'model_9.pth'))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device).eval()

    # Define the loss function with no reduction to compute loss per sample
    loss_fn = nn.MSELoss(reduction='none')

    # Calculate threshold based on training data
    loss_list = []
    with torch.no_grad():
        for x_batch, _ in tqdm(train_loader, desc='Processing training data'):
            x_batch = x_batch.to(device)

            # Forward pass
            x_reconstruction, _, _ = model(x_batch)

            # Compute and accumulate loss for each sample
            cmp_factor = loss_fn(x_batch, x_reconstruction).mean(dim=[1])
            loss_list.extend(cmp_factor.cpu().numpy())  # Convert to numpy for faster concatenation

    # Set threshold at the 80th percentile
    threshold = np.quantile(loss_list, 0.8)

    # Test phase: Initialize lists for comparison factors and labels
    comparison_factors = []
    labels = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc='Testing'):
            x_batch = x_batch.to(device)

            # Forward pass
            x_reconstruction, _, _ = model(x_batch)

            # Compute loss for each sample
            cmp_factor = loss_fn(x_batch, x_reconstruction).mean(dim=[1])
            comparison_factors.extend(cmp_factor.cpu().numpy())  # Convert to numpy
            labels.extend(y_batch.numpy())  # Convert to numpy for faster ops

    # Generate predictions using the threshold
    predictions = (np.array(comparison_factors) > threshold).astype(int)

    # Calculate and return performance metrics
    return calculate_metrics(labels, predictions)


if __name__ == "__main__":
    metrics = test(cfg)
    print(metrics)
