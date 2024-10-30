from model import VAE_CNN
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set the configuration
cfg = {
    'gpu_usage': True,
    'seed': 42,
    'train_dataset_dir': 'data/train',
    'val_dataset_dir': 'data/val',
    'batch_size': 32,
    'num_workers': 4,
    'learning_rate': 1e-3,
    'n_epochs': 20,
    'save_dir': 'checkpoints',
    'log_dir': 'runs',  # Directory for TensorBoard logs
}


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


def calculate_threshold(epoch_losses):
    # Calculate the threshold (T80) based on the 80th percentile of epoch losses
    return torch.quantile(torch.tensor(epoch_losses), 0.8).item()


def train(cfg):
    device = torch.device('cuda' if cfg['gpu_usage'] else 'cpu')

    # Load data
    transform_train = transforms.Compose([
        transforms.Resize(100),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(datasets.ImageFolder(cfg['train_dataset_dir'], transform=transform_train),
                              batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(datasets.ImageFolder(cfg['val_dataset_dir'], transform=transform_val),
                            batch_size=cfg['batch_size'])

    # Create the AutoEncoder model
    model = VAE_CNN().to(device)
    model.train()  # Set model to training mode

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'])

    # Define the loss function
    loss_fn = customLoss()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=cfg['log_dir'])

    # Training loop
    n_epochs = cfg['n_epochs']
    for epoch in range(n_epochs):
        print(f"\nTraining epoch {epoch + 1}/{n_epochs}")
        print("-----------------------------------")

        # Training phase
        model.train()
        pbar = tqdm(train_loader, ncols=80, desc='Training')
        running_loss = 0.0
        train_num = 0
        for step, x in enumerate(pbar):
            x = x[0].to(device)  # Move input to device

            # Clear the old gradients
            optimizer.zero_grad()

            # Compute the forward pass
            recon_batch, mu, logvar = model(x)

            # Compute loss and update weights
            loss = loss_fn(recon_batch, x, mu, logvar)
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

            # Aggregate loss
            running_loss += loss.item() * x.shape[0]
            train_num += x.shape[0]

        # Calculate and print average loss for the epoch
        train_loss = running_loss / train_num
        print(f"\tTrain loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_num = 0
        with torch.no_grad():
            for x in val_loader:
                x = x[0].to(device)  # Move input to device

                # Compute the forward pass
                x_reconstruction, mu, logvar = model(x)

                # Compute loss
                val_loss = loss_fn(x, x_reconstruction, mu, logvar)

                # Aggregate validation loss
                val_running_loss += val_loss.item() * x.shape[0]
                val_num += x.shape[0]

        # Calculate and print average validation loss for the epoch
        val_loss = val_running_loss / val_num
        print(f"\tValidation loss: {val_loss:.4f}")

        # Log average train and validation loss per epoch to TensorBoard
        writer.add_scalar('Loss/Train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/Validation_epoch', val_loss, epoch)

        # Save model checkpoint
        torch.save(model.state_dict(), f"{cfg['save_dir']}/model_{epoch}.pth")

    writer.close()  # Close the writer after training is done


if __name__ == "__main__":
    train(cfg)
