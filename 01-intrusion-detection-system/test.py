import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn import metrics
from model import AutoEncoder
from process_data import prepare_data
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration settings
cfg = {
    'gpu_usage': True,
    'seed': 42,
    'dataset_dir': 'data/UNSW',
    'test_batch_size': 8,
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

    # Load test data
    X_test, y_test = prepare_data(dir_path=cfg['dataset_dir'], data_type='test')

    # Convert data to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # Create DataLoader for the test set
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)

    # Initialize the AutoEncoder model
    model = AutoEncoder(n_features=X_test.shape[1])

    # Load the saved model checkpoint
    model_path = os.path.join(cfg['save_dir'], 'model_9.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    # Define the loss function with no reduction to compute loss per sample
    loss_fn = nn.MSELoss(reduction='none')

    # Load the training losses to determine the threshold
    train_losses = np.loadtxt(os.path.join(cfg['save_dir'], 'epoch_losses.txt'))
    threshold = train_losses[-1]  # Use the last epoch's loss as the threshold

    # Initialize lists for comparison factors and labels
    comparison_factors = []
    labels = []

    # Test loop
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc='Testing'):
            x_batch = x_batch.to(device)

            # Forward pass through the model
            _, x_reconstruction = model(x_batch)

            # Compute the loss for each sample in the batch
            cmp_factor = loss_fn(x_batch, x_reconstruction).mean(dim=1)
            comparison_factors.extend(cmp_factor.cpu().tolist())
            labels.extend(y_batch.tolist())

    # Generate predictions based on the threshold
    predictions = [int(factor > threshold) for factor in comparison_factors]

    # Calculate performance metrics
    performance_metrics = calculate_metrics(labels, predictions)

    return performance_metrics


if __name__ == "__main__":
    metrics = test(cfg)
    print(metrics)
