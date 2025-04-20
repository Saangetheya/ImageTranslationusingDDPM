import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
import random
from scipy.ndimage import zoom

def load_data(path):
    scan = nib.load(path)
    volume = scan.get_fdata()
    min = np.amax(volume)
    max = np.amin(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_3d(volume, target_shape):
    """
    Resize a 3D volume to target shape using scipy.ndimage.zoom
    
    Args:
        volume (np.ndarray): Input 3D volume of shape [H, W, L]
        target_shape (tuple): Target shape [h, w, l]
        
    Returns:
        np.ndarray: Resized volume of shape [h, w, l]
    """
    # Calculate zoom factors for each dimension
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    
    # Apply zoom
    resized_volume = zoom(volume, zoom_factors, order=3)  # order=3 for cubic interpolation
    
    return resized_volume

class MonaiMRIDataset(Dataset):
    def __init__(self, root_dir, index_file, image_size):
            
        with open(os.path.join(root_dir, index_file), 'r') as f:
            self.file_paths = [line.strip() for line in f if line.strip()]
        
        self.image_size = image_size
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        item = self.file_paths[idx]
        t1_image_path = os.path.join(self.root_dir, item[1])
        dwi_image_path = os.path.join(self.root_dir, item[2])

        t1_image = load_data(t1_image_path)
        dwi_image = load_data(dwi_image_path)

        t1_image = resize_3d(t1_image, self.image_size)
        dwi_image = resize_3d(dwi_image, self.image_size)

        diagnosis = random.randint(0, 2)

        return {"t1_image": t1_image, "dwi_image": dwi_image, "diagnosis": diagnosis}
