import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random


class Brain3DDataset(Dataset):
    """
    Load consecutive MRI slices to form 3D temporal data
    """

    def __init__(self, data_dir, temporal_length=5):
        """
        Args:
            data_dir (str): Directory containing the data files
            temporal_length (int): Number of consecutive slices to load for temporal dimension
        """
        self.data_dir = data_dir
        self.temporal_length = temporal_length

        # Load index file
        txt_file = os.path.join(data_dir, 'index.txt')
        with open(txt_file, 'r') as f:
            all_indices = [line.strip() for line in f.readlines()]

        # Organize slices by direction (sagittal/coronal/axial) and subject ID
        self.slice_groups = {}
        for line in all_indices:
            parts = line.split(' ')
            subject_id = parts[0]
            t1_path = parts[1]
            dwi_path = parts[2]

            # Extract direction and index from path
            # Assuming format: T1/subject_id_direction_idx.png
            t1_filename = os.path.basename(t1_path)
            direction_idx = t1_filename.split('_')[1:]  # ['direction', 'idx.png']
            direction = direction_idx[0]
            slice_idx = int(direction_idx[1].split('.')[0])

            key = f"{subject_id}_{direction}"
            if key not in self.slice_groups:
                self.slice_groups[key] = []

            self.slice_groups[key].append({
                'subject_id': subject_id,
                'direction': direction,
                'slice_idx': slice_idx,
                't1_path': t1_path,
                'dwi_path': dwi_path
            })

        # Sort slices within each group by index
        for key in self.slice_groups:
            self.slice_groups[key] = sorted(self.slice_groups[key], key=lambda x: x['slice_idx'])

        # Create valid sequences (consecutive temporal_length slices)
        self.valid_sequences = []
        for key, slices in self.slice_groups.items():
            if len(slices) >= temporal_length:
                for i in range(len(slices) - temporal_length + 1):
                    self.valid_sequences.append((key, i))

    def __len__(self):
        """Return the total number of sequences in the dataset"""
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        """
        Get a 3D sequence for the specified index

        Args:
            idx (int): Sequence index

        Returns:
            dict: Dictionary containing T1, DWI 3D sequence data and metadata
        """
        key, start_idx = self.valid_sequences[idx]
        group_slices = self.slice_groups[key]
        sequence_slices = group_slices[start_idx:start_idx + self.temporal_length]

        t1_sequence = []
        dwi_sequence = []

        # Generate random diagnosis ID to match 2D version
        diagnosis = random.randint(0, 2)

        for slice_info in sequence_slices:
            t1_path = os.path.join(self.data_dir, slice_info['t1_path'])
            dwi_path = os.path.join(self.data_dir, slice_info['dwi_path'])

            # Read images
            t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
            dwi_img = cv2.imread(dwi_path, cv2.IMREAD_GRAYSCALE)

            # Center crop and resize
            h, w = t1_img.shape
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2

            t1_img = t1_img[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]
            dwi_img = dwi_img[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]

            # Resize to standard size
            t1_img = cv2.resize(t1_img, (64, 64), interpolation=cv2.INTER_LINEAR)
            dwi_img = cv2.resize(dwi_img, (64, 64), interpolation=cv2.INTER_LINEAR)

            # Normalize and convert to tensors
            t1_img = 1 - torch.from_numpy(t1_img).float() / 255.0
            dwi_img = 1 - torch.from_numpy(dwi_img).float() / 255.0

            # Add channel dimension
            if len(t1_img.shape) == 2:
                t1_img = t1_img.unsqueeze(0) * 2 - 1
            if len(dwi_img.shape) == 2:
                dwi_img = dwi_img.unsqueeze(0) * 2 - 1

            t1_sequence.append(t1_img)
            dwi_sequence.append(dwi_img)

        # Stack slices to create 3D tensor [C, T, H, W]
        t1_volume = torch.stack(t1_sequence, dim=1)
        dwi_volume = torch.stack(dwi_sequence, dim=1)

        # Extract direction and subject ID
        subject_id, direction = key.split('_')

        return {
            'T1': t1_volume,
            'DWI': dwi_volume,
            'subject_id': subject_id,
            'direction': direction,
            'start_idx': start_idx,
            # 'diagnosis': diagnosis,  # Return random diagnosis ID to match 2D version
            'diagnosis': torch.tensor(diagnosis, dtype=torch.long)
        }