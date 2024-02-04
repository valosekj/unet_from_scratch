import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class SpinalCordDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        """
        Args:
            image_path (string): Path to the NIfTI file with spinal cord MRI images.
            mask_path (string): Path to the NIfTI file with corresponding masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image = self.load_nifti(image_path)
        self.mask = self.load_nifti(mask_path)
        self.transform = transform

    def load_nifti(self, file_path):
        """
        Load a NIfTI file
        Args:
            file_path (string): Path to the NIfTI file.
        Returns:
            numpy.ndarray: Image data in RPI orientation.
        """
        img = nib.load(file_path)
        data = img.get_fdata()
        return data

    def __len__(self):
        return self.image.shape[2]  # Assuming axial slices are along the 3rd dimension

    def __getitem__(self, idx):
        image_slice = self.image[:, :, idx]
        mask_slice = self.mask[:, :, idx]

        #print(f'{image_slice.shape}')

        # Add channel dimension for PyTorch, shape becomes [height, width, channels]
        image_slice = np.expand_dims(image_slice, axis=-1)
        mask_slice = np.expand_dims(mask_slice, axis=-1)

        #print(f'{image_slice.shape}')

        # Create a copy of the image_slice and mask_slice for visualization
        image_slice_copy = image_slice.copy()
        mask_slice_copy = mask_slice.copy()

        # Apply transforms if any
        if self.transform:
            image_slice = self.transform(image_slice)
            mask_slice = self.transform(mask_slice)

        #print(f'{image_slice.shape}')

        # # Plot the image_slice_copy, mask_slice_copy, image_slice, and mask_slice
        # fig, axes = plt.subplots(2, 2)
        # axes[0, 0].imshow(image_slice_copy.squeeze(), cmap='gray')
        # axes[0, 0].set_title('Original Image')
        # axes[0, 1].imshow(mask_slice_copy.squeeze(), cmap='gray')
        # axes[0, 1].set_title('Original Mask')
        # axes[1, 0].imshow(image_slice.squeeze(), cmap='gray')
        # axes[1, 0].set_title('Transformed Image')
        # axes[1, 1].imshow(mask_slice.squeeze(), cmap='gray')
        # axes[1, 1].set_title('Transformed Mask')
        # plt.tight_layout()
        # plt.show()

        # Convert to float to match the data type of the PyTorch weights and biases
        image_slice = image_slice.type(torch.FloatTensor)
        mask_slice = mask_slice.type(torch.FloatTensor)

        return image_slice, mask_slice
