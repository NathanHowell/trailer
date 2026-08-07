import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from diffusers import UNet2DModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import glob


class DEMTrailDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset for DEM and hiking trail data.
        Args:
            data_dir (str): Directory with numpy files containing DEM and trail data
            transform: Optional transforms to apply
        """
        self.data_files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load combined data (assuming each file has shape [2, H, W])
        # with channel 0 = DEM, channel 1 = trail mask
        data = np.load(self.data_files[idx])

        # Split into input and target
        dem = data[0:1]  # Keep channel dimension
        trail = data[1:2]

        # Convert to tensors
        dem_tensor = torch.from_numpy(dem).float()
        trail_tensor = torch.from_numpy(trail).float()

        if self.transform:
            dem_tensor = self.transform(dem_tensor)
            trail_tensor = self.transform(trail_tensor)

        return dem_tensor, trail_tensor


def train(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders
    train_dataset = DEMTrailDataset(config["train_data_dir"])
    val_dataset = DEMTrailDataset(config["val_data_dir"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"])

    # Initialize model
    model = UNet2DModel(
        sample_size=config["image_size"],  # Image size
        in_channels=1,  # DEM channel
        out_channels=1,  # Trail prediction
        layers_per_block=2,  # Number of layers per block
        block_out_channels=(64, 128, 256, 512),  # Channel dimensions
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0

        for dem, trail_mask in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            dem = dem.to(device)
            trail_mask = trail_mask.to(device)

            # Forward pass
            outputs = model(dem, timestep=0).sample
            loss = criterion(outputs, trail_mask)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for dem, trail_mask in tqdm(val_loader, desc="Validation"):
                dem = dem.to(device)
                trail_mask = trail_mask.to(device)

                outputs = model(dem, timestep=0).sample
                loss = criterion(outputs, trail_mask)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(config["checkpoint_dir"], "best_model.pth"))

        # Save example predictions
        if epoch % config["save_every"] == 0:
            save_predictions(model, val_loader, device, epoch, config["output_dir"])


def save_predictions(model, dataloader, device, epoch, output_dir):
    """Save sample predictions for visualization"""
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Get a single batch
        dem, trail_mask = next(iter(dataloader))
        dem = dem.to(device)

        # Generate predictions
        outputs = model(dem, timestep=0).sample
        predictions = torch.sigmoid(outputs) > 0.5

        # Save a few examples
        for i in range(min(5, len(dem))):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # DEM
            axs[0].imshow(dem[i, 0].cpu().numpy(), cmap='terrain')
            axs[0].set_title("DEM")
            axs[0].axis('off')

            # Ground truth trail
            axs[1].imshow(trail_mask[i, 0].cpu().numpy(), cmap='gray')
            axs[1].set_title("Ground Truth Trail")
            axs[1].axis('off')

            # Predicted trail
            axs[2].imshow(predictions[i, 0].cpu().numpy(), cmap='gray')
            axs[2].set_title("Predicted Trail")
            axs[2].axis('off')

            plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_sample_{i}.png"))
            plt.close(fig)


def main():
    config = {
        "train_data_dir": "data/train",
        "val_data_dir": "data/val",
        "output_dir": "outputs",
        "checkpoint_dir": "checkpoints",
        "batch_size": 8,
        "num_workers": 4,
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "image_size": 256,
        "save_every": 5,
    }

    # Create directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Start training
    train(config)


if __name__ == "__main__":
    main()