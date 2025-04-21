import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class Brain3DDataset(Dataset):
    """
    加载连续的MRI切片以形成3D时间序列数据
    """

    def __init__(self, data_dir, temporal_length=5):
        """
        Args:
            data_dir (str): 包含数据文件的目录
            temporal_length (int): 要加载的连续切片数量，表示时间维度
        """
        self.data_dir = data_dir
        self.temporal_length = temporal_length

        # 加载索引文件
        txt_file = os.path.join(data_dir, 'index.txt')
        with open(txt_file, 'r') as f:
            all_indices = [line.strip() for line in f.readlines()]

        # 按方向(sagittal/coronal/axial)和主题ID组织切片
        self.slice_groups = {}
        for line in all_indices:
            parts = line.split(' ')
            subject_id = parts[0]
            t1_path = parts[1]
            dwi_path = parts[2]

            # 从路径中提取方向和索引
            # 假设格式为: T1/subject_id_direction_idx.png
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

        # 对每个组内的切片按索引排序
        for key in self.slice_groups:
            self.slice_groups[key] = sorted(self.slice_groups[key], key=lambda x: x['slice_idx'])

        # 创建有效序列（连续的temporal_length个切片）
        self.valid_sequences = []
        for key, slices in self.slice_groups.items():
            if len(slices) >= temporal_length:
                for i in range(len(slices) - temporal_length + 1):
                    self.valid_sequences.append((key, i))

    def __len__(self):
        """返回数据集中的序列总数"""
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        """
        获取指定索引的3D序列

        Args:
            idx (int): 序列索引

        Returns:
            dict: 包含T1, DWI 3D序列数据和元信息
        """
        key, start_idx = self.valid_sequences[idx]
        group_slices = self.slice_groups[key]
        sequence_slices = group_slices[start_idx:start_idx + self.temporal_length]

        t1_sequence = []
        dwi_sequence = []

        for slice_info in sequence_slices:
            t1_path = os.path.join(self.data_dir, slice_info['t1_path'])
            dwi_path = os.path.join(self.data_dir, slice_info['dwi_path'])

            # 读取图像
            t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
            dwi_img = cv2.imread(dwi_path, cv2.IMREAD_GRAYSCALE)

            # 中心裁剪和调整大小
            h, w = t1_img.shape
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2

            t1_img = t1_img[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]
            dwi_img = dwi_img[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]

            # 调整为标准大小
            t1_img = cv2.resize(t1_img, (64, 64), interpolation=cv2.INTER_LINEAR)
            dwi_img = cv2.resize(dwi_img, (64, 64), interpolation=cv2.INTER_LINEAR)

            # 归一化和转换为张量
            t1_img = 1 - torch.from_numpy(t1_img).float() / 255.0
            dwi_img = 1 - torch.from_numpy(dwi_img).float() / 255.0

            # 添加通道维度
            if len(t1_img.shape) == 2:
                t1_img = t1_img.unsqueeze(0) * 2 - 1
            if len(dwi_img.shape) == 2:
                dwi_img = dwi_img.unsqueeze(0) * 2 - 1

            t1_sequence.append(t1_img)
            dwi_sequence.append(dwi_img)

        # 堆叠切片以创建3D张量 [C, T, H, W]
        t1_volume = torch.stack(t1_sequence, dim=1)
        dwi_volume = torch.stack(dwi_sequence, dim=1)

        # 获取方向和主题ID信息
        subject_id, direction = key.split('_')

        return {
            'T1': t1_volume,
            'DWI': dwi_volume,
            'subject_id': subject_id,
            'direction': direction,
            'start_idx': start_idx
        }



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
        subject_id = item[0]
        
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
            'subject_id': subject_id
        }