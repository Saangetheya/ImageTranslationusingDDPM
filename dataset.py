import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class BrainDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Directory containing the data files
            txt_file (str): Path to the txt file containing dataset indices
        """
        self.data_dir = data_dir
        txt_file = os.path.join(data_dir, 'index.txt')
        with open(txt_file, 'r') as f:
            self.indices = [line.strip() for line in f.readlines()]
            
    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return
            
        Returns:
            dict: Dictionary containing T1, DWI and subject_id
        """
        item = self.indices[idx]
        item = item.split(' ')

        t1_path = os.path.join(self.data_dir,item[1])
        dwi_path = os.path.join(self.data_dir,item[2])
        diagnosis = random.randint(0, 2)
        
        t1_data = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
        dwi_data = cv2.imread(dwi_path, cv2.IMREAD_GRAYSCALE)

        # Get current dimensions
        h, w = t1_data.shape
        
        # Calculate dimensions for center crop
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        
        # Perform center crop
        t1_data = t1_data[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]
        dwi_data = dwi_data[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]
        
        # Resize to 64x64
        t1_data = cv2.resize(t1_data, (64, 64), interpolation=cv2.INTER_LINEAR)
        dwi_data = cv2.resize(dwi_data, (64, 64), interpolation=cv2.INTER_LINEAR)

        t1_data = 1 - torch.from_numpy(t1_data).float() / 255.0
        dwi_data = 1 - torch.from_numpy(dwi_data).float() / 255.0

        if len(t1_data.shape) == 2:
            t1_data = t1_data.unsqueeze(0) * 2 - 1
        if len(dwi_data.shape) == 2:
            dwi_data = dwi_data.unsqueeze(0) * 2 - 1
        
        return {
            'T1': t1_data,
            'DWI': dwi_data,
            'diagnosis': diagnosis
        }
