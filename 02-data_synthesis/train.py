from model import Generator, Discriminator
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image


# Set the configuration
cfg = {
    'gpu_usage': True,
    'seed': 42,
    'dataset_dir': 'data',
    'train_batch_size': 16,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'betas': (0.5, 0.999),
    'n_epochs': 20,
    'save_dir': 'checkpoints',
    'log_dir': 'runs',  # Directory for TensorBoard logs
    'image_size': 256,
    'label_dim': 9,
    'G_input_dim': 100,
    'G_output_dim': 3,
    'D_input_dim': 3,
    'D_output_dim': 3,
    'num_filters': [1024, 512, 256, 128, 64, 32],
}


class MalwareDataset(Dataset):
    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def read_data_set(self):
        all_img_files = []
        all_labels = []
        class_names = os.walk(self.data_set_path).__next__()[1]
        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)
        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length


def train(cfg):
    device = torch.device('cuda' if cfg['gpu_usage'] else 'cpu')
    transform_train = transforms.Compose([transforms.Resize((cfg['image_size'], cfg['image_size'])),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_dataset = MalwareDataset(os.path.join(cfg['dataset_dir'], 'train'), transform_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'])

    generator_model = Generator(cfg['G_input_dim'], cfg['label_dim'], cfg['num_filters'], cfg['G_output_dim'])
    discriminator_model = Discriminator(cfg['D_input_dim'], cfg['label_dim'], cfg['num_filters'][::-1], cfg['D_output_dim'])
    generator_model = generator_model.to(device)
    discriminator_model = discriminator_model.to(device)
    generator_model.train()
    discriminator_model.train()

    G_optimizer = Adam(generator_model.parameters(), lr=cfg['learning_rate'], betas=cfg['betas'])
    D_optimizer = Adam(discriminator_model.parameters(), lr=cfg['learning_rate'] * 0.01, betas=cfg['betas'])

    # Define the loss function
    loss_fn = torch.nn.BCELoss()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=cfg['log_dir'])

    # label preprocess
    onehot = torch.zeros(cfg['label_dim'], cfg['label_dim'])
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8]).view(cfg['label_dim'], 1), 1).view(cfg['label_dim'], cfg['label_dim'], 1, 1)
    fill = torch.zeros([cfg['label_dim'], cfg['label_dim'], cfg['image_size'], cfg['image_size']])
    for dim in range(cfg['label_dim']):
        fill[dim, dim, :, :] = 1

    # Training loop
    n_epochs = cfg['n_epochs']
    for epoch in range(n_epochs):
        print(f"\nTraining epoch {epoch + 1}/{n_epochs}")
        print("-----------------------------------")

        if epoch % 5:
            G_optimizer.param_groups[0]['lr'] /= 2
        if epoch % 10:
            G_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        pbar = tqdm(train_loader, ncols=80, desc='Training')
        D_running_loss = 0.0
        G_running_loss = 0.0
        train_num = 0
        for step, data in enumerate(pbar):
            images, labels = data['image'], data['label']
            images = images.to(device)
            labels = labels.to(device)
            mini_batch = images.size()[0]
            labels_fill_ = fill[labels].to(device)

            # Train discriminator with real data
            D_real_decision = discriminator_model(images, labels_fill_).squeeze()
            y_real_ = torch.ones_like(D_real_decision).to(device)
            y_fake_ = torch.zeros_like(D_real_decision).to(device)
            D_real_loss = loss_fn(D_real_decision, y_real_)

            # Train discriminator with fake data
            z_ = torch.randn(mini_batch, cfg['G_input_dim']).view(-1, cfg['G_input_dim'], 1, 1).to(device)
            c_ = (torch.rand(mini_batch, 1) * cfg['label_dim']).type(torch.LongTensor).squeeze()
            if len(c_.shape) == 0:
                c_ = c_.unsqueeze(0)
            c_onehot_ = onehot[c_].to(device)
            gen_image = generator_model(z_, c_onehot_)

            c_fill_ = fill[c_].cuda()
            D_fake_decision = discriminator_model(gen_image, c_fill_).squeeze()
            D_fake_loss = loss_fn(D_fake_decision, y_fake_)

            D_loss = D_real_loss + D_fake_loss
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # Train generator
            z_ = torch.randn(mini_batch, cfg['G_input_dim']).view(-1, cfg['G_input_dim'], 1, 1).to(device)
            c_ = (torch.rand(mini_batch, 1) * cfg['label_dim']).type(torch.LongTensor).squeeze()
            if len(c_.shape) == 0:
                c_ = c_.unsqueeze(0)
            c_onehot_ = onehot[c_].to(device)
            gen_image = generator_model(z_, c_onehot_)

            c_fill_ = fill[c_].cuda()
            D_fake_decision = discriminator_model(gen_image, c_fill_).squeeze()
            G_loss = loss_fn(D_fake_decision, y_real_)

            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            D_running_loss += D_loss.item() * mini_batch
            G_running_loss += G_loss.item() * mini_batch
            train_num += mini_batch

        # Calculate and print average loss for the epoch
        D_train_loss = D_running_loss / train_num
        G_train_loss = G_running_loss / train_num
        print(f"\tG train loss: {G_train_loss:.4f}, D_train_loss: {D_train_loss:.4f}")

        # Log average train and validation loss per epoch to TensorBoard
        writer.add_scalar('Discriminator Loss/Train_epoch', D_train_loss, epoch)
        writer.add_scalar('Generator Loss/Train_epoch', G_train_loss, epoch)

        # Save model checkpoint
        torch.save(generator_model.state_dict(), f"{cfg['save_dir']}/G_model_{epoch}.pth")
        torch.save(discriminator_model.state_dict(), f"{cfg['save_dir']}/D_model_{epoch}.pth")

    writer.close()  # Close the writer after training is done


if __name__ == "__main__":
    train(cfg)
