from model import AutoEncoder
from process_data import prepare_data
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Set the configuration
cfg = {
    'gpu_usage': True,
    'seed': 42,
    'dataset_dir': 'data/UNSW',
    'train_batch_size': 8,
    'num_workers': 4,
    'learning_rate': 1e-3,
    'n_epochs': 10,
    'save_dir': 'checkpoints',
    'log_dir': 'runs/autoencoder_experiment',  # Directory for TensorBoard logs
    'validation_split': 0.2  # Fraction of data to use for validation
}


def train(cfg):
    device = torch.device('cuda' if cfg['gpu_usage'] else 'cpu')

    # Load and prepare data
    X_train, _ = prepare_data(dir_path=cfg['dataset_dir'], data_type='train')

    # Convert numpy arrays to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)

    # Split the data into training and validation sets
    val_size = int(cfg['validation_split'] * len(X_train))
    train_size = len(X_train) - val_size
    train_dataset, val_dataset = random_split(X_train, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['train_batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'])

    # Create the AutoEncoder model
    model = AutoEncoder(n_features=X_train.shape[1])
    model = model.to(device)
    model.train()  # Set model to training mode

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'])

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=cfg['log_dir'])

    # Training loop
    n_epochs = cfg['n_epochs']
    epoch_losses = []
    for epoch in range(n_epochs):
        print(f"\nTraining epoch {epoch + 1}/{n_epochs}")
        print("-----------------------------------")

        # Training phase
        model.train()
        pbar = tqdm(train_loader, ncols=80, desc='Training')
        running_loss = 0.0
        train_num = 0
        for step, x in enumerate(pbar):  # Tuple unpacking (x,)
            x = x.to(device)  # Move input to device

            # Clear the old gradients
            optimizer.zero_grad()

            # Compute the forward pass
            _, x_reconstruction = model(x)

            # Compute loss and update weights
            loss = loss_fn(x, x_reconstruction)
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
                _, x_reconstruction = model(x)

                # Compute loss
                val_loss = loss_fn(x, x_reconstruction)

                # Aggregate validation loss
                val_running_loss += val_loss.item() * x.shape[0]
                val_num += x.shape[0]

        # Calculate and print average validation loss for the epoch
        val_loss = val_running_loss / val_num
        print(f"\tValidation loss: {val_loss:.4f}")

        # Log average train and validation loss per epoch to TensorBoard
        writer.add_scalar('Loss/Train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/Validation_epoch', val_loss, epoch)

        # Append loss to the list
        epoch_losses.append(train_loss)

        # Save model checkpoint
        torch.save(model.state_dict(), f"{cfg['save_dir']}/model_{epoch}.pth")

    writer.close()  # Close the writer after training is done

    # Write the list of epoch losses to a file
    with open(f"{cfg['save_dir']}/epoch_losses.txt", 'w') as f:
        for loss in epoch_losses:
            f.write(f"{loss}\n")


if __name__ == "__main__":
    train(cfg)
