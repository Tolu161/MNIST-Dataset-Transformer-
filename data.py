import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sentencepiece as spm

class MNISTGridDataset(Dataset):
    def __init__(self, mnist_data, num_samples=60000, patch_size=14, embedding_dim=64):
        self.mnist_data = mnist_data
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # Extract images and labels from MNIST dataset
        self.images = mnist_data.data.unsqueeze(1).float()  # Shape: [N, 1, 28, 28]
        self.labels = mnist_data.targets  # Shape: [N]

        # Generate grids and patches
        self.patches, self.grid_labels = self.create_grids_and_patches()

        # Define a linear layer for embedding generation
        self.projection = nn.Linear(patch_size * patch_size, embedding_dim)

    def create_grids_and_patches(self):
        # Randomly select 4 images and labels to create a 2x2 grid
        indices = torch.randint(0, len(self.images), (self.num_samples, 4))
        selected_images = self.images[indices]  # Shape: [num_samples, 4, 1, 28, 28]
        grid_labels = self.labels[indices]  # Shape: [num_samples, 4]

        # Reshape to create 2x2 grids
        grids = selected_images.view(self.num_samples, 2, 2, 1, 28, 28)
        grids = torch.cat([
            torch.cat([grids[:, 0, 0], grids[:, 0, 1]], dim=3),  # Top row
            torch.cat([grids[:, 1, 0], grids[:, 1, 1]], dim=3)   # Bottom row
        ], dim=2)  # Shape: [num_samples, 1, 56, 56]

        # Split grids into patches using unfold
        patches = grids.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(self.num_samples, -1, 1, self.patch_size, self.patch_size)  # Shape: [num_samples, 16, 1, 14, 14]

        return patches, grid_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        patches = self.patches[idx]  # Shape: [16, 1, 14, 14]
        grid_labels = self.grid_labels[idx]  # Shape: [4]

        # Flatten patches and generate embeddings
        patches_flat = patches.view(patches.size(0), -1)  # Shape: [16, 196]
        embeddings = self.projection(patches_flat)  # Shape: [16, embedding_dim]

        # Tokenize labels using SentencePiece
        caption = " ".join(str(digit.item()) for digit in grid_labels)
        caption_input = self.sp.encode_as_ids("<s> " + caption)  # Add start token
        caption_label = self.sp.encode_as_ids(caption + " </s>")  # Add end token

        return {
            "embeddings": embeddings,  # Shape: [16, embedding_dim]
            "caption_input": torch.tensor(caption_input),  # Tokenized input
            "caption_label": torch.tensor(caption_label),  # Tokenized label
            "labels": grid_labels  # Original labels
        }

def get_dataloader(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    dataset = MNISTGridDataset(mnist_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader