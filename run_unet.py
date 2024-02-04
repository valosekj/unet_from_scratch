import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

import matplotlib.pyplot as plt

from data_loading import SpinalCordDataset
from unet_blocks import UNet
from losses import dice_loss


BATCHE_SIZE = 8


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer):
    """
    This function trains the model for a given number of epochs, and at each
    epoch, it processes the training data in batches, updates the model parameters,
    and evaluates the performance on the validation dataset. The training and
    validation losses are computed using a combination of Binary Cross Entropy
    (BCE) loss and Dice loss.

    Parameters:
    - model: A PyTorch model instance that will be trained.
    - train_loader: A DataLoader instance containing the training data.
    - val_loader: A DataLoader instance containing the validation data.
    - epochs: An integer representing the number of epochs for training.
    - criterion: The loss function used for training.
    - optimizer: The optimization algorithm used to update the model's weights.
    """
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        # Enumerate over the training batches consisting of individual MRI slices
        # The number of batches = number train samples / batch_size
        # Each batch (`len(images)`) consists of 8 slices (BATCH_SIZE)
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            # plt.imshow(images[1,:,:,:].squeeze())
            optimizer.zero_grad()

            # Through the training process, the model learns to minimize this loss. In other words, it learns to
            # generate predictions (outputs) that are as close as possible to the true masks (masks).
            outputs = model(images)

            # Sanity plot -- plot the first image in the batch, the mask, and the output
            if i == 0:
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                # Rotate the image by 90 degrees
                axes[0].imshow(np.rot90(images[0, :, :, :].squeeze()), cmap='gray')
                axes[0].set_title('Image')
                axes[1].imshow(np.rot90(outputs[0, :, :, :].detach().numpy().squeeze()), cmap='gray')
                axes[1].set_title('Output')
                axes[2].imshow(np.rot90(masks[0, :, :, :].squeeze()), cmap='gray')
                axes[2].set_title('Mask')
                plt.tight_layout()
                # Save the image
                plt.savefig(f'images_training/output_epoch{epoch}_batch{i+1}.png', dpi=200)
                print(f'Image saved: images_training/output_epoch{epoch}_batch{i+1}.png')

            # Calculate the combined loss
            bce_loss = criterion(outputs, masks)
            d_loss = dice_loss(outputs, masks)
            loss = bce_loss + d_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f'Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item()}')

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images)

                # Calculate the combined loss for validation data
                bce_loss = criterion(outputs, masks)
                d_loss = dice_loss(outputs, masks)
                loss = bce_loss + d_loss

                val_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, '
              f'Validation Loss: {val_loss/len(val_loader)}')

# -----------------------------
# Load the spinal cord dataset
# -----------------------------
# Get path to this repo
path_repo = os.path.dirname(os.path.abspath(__file__))

# T2w MRI
# https://github.com/spine-generic/data-multi-subject/blob/master/sub-amu01/anat/sub-amu01_T2w.nii.gz
image_path = os.path.join(path_repo, 'data', 'sub-amu01_T2w.nii.gz')
# Binary mask of the spinal cord
# https://github.com/spine-generic/data-multi-subject/blob/master/derivatives/labels/sub-amu01/anat/sub-amu01_T2w_seg-manual.nii.gz
mask_path = os.path.join(path_repo, 'data', 'sub-amu01_T2w_seg-manual.nii.gz')

# Just some example transforms; nothing fancy
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
#    transforms.RandomHorizontalFlip(),
#    transforms.RandomRotation(10),
#    transforms.Normalize(mean=[0.5], std=[0.5])
])

spinal_dataset = SpinalCordDataset(image_path, mask_path, transform=transform)

# Parameters for dataset split
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits
dataset_size = len(spinal_dataset)  # Get the number of samples (axial slices)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

# This shuffling is done before splitting the dataset into training and validation sets. It ensures that the split
# between training and validation data is random.
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Sanity check -- check the image and mask
# fig, axes = plt.subplots(1, 2)
# # Get the image and mask
# image_slice, mask_slice = spinal_dataset[train_indices[0]]
# # Remove the channel dimension (slices are tensors)
# image_slice = image_slice.squeeze()
# mask_slice = mask_slice.squeeze()
# axes[0].imshow(image_slice, cmap='gray')
# axes[1].imshow(mask_slice, cmap='gray')
# plt.show()

# Creating data samplers and loaders (number of training and validation samples (slices))
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
print(f'Length of train_sampler (i.e., number of training slices): {len(train_sampler)}')
print(f'Length of validation_sampler (i.e., number of validation slices): {len(validation_sampler)}')

# Create data loaders
# - `SubsetRandomSampler` randomly samples from the provided indices, ensuring that each epoch sees a different
# permutation of the dataset. This randomness is beneficial for generalizing the model and preventing overfitting.
# - The batch_size parameter specifies the number of samples that will be passed through the network at one time.
# `batch_size=2` means that eight slices from the spinal cord MRI dataset will be processed together in each step of
# the training or inference process.
train_loader = DataLoader(spinal_dataset, batch_size=BATCHE_SIZE, sampler=train_sampler)
validation_loader = DataLoader(spinal_dataset, batch_size=BATCHE_SIZE, sampler=validation_sampler)
print(f'BATCH_SIZE: {BATCHE_SIZE}')
# The length of the train_loader = train_sampler / batch_size
print(f'Length of train_loader (i.e., number of training batches): {len(train_loader)}')
print(f'Length of validation_loader (i.e., number of validation batches): {len(validation_loader)}')

# Define a model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
# Use the sum of the binary cross-entropy loss and the dice loss as the loss function
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
train_model(model, train_loader, validation_loader, epochs=25, criterion=criterion, optimizer=optimizer)

