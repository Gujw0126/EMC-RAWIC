import rawpy
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset
import random
import csv
from PIL import Image
    
#samsung, olympus testset
class NUSFolder(Dataset):
    def __init__(self, root, split='train', crop='random', patch_size=512, bit_depth=14):
        self.samples = []
        dir_path = os.path.join(root, split)
        for filename in os.listdir(dir_path):
            self.samples.append(os.path.join(dir_path, filename))
        self.crop = crop
        self.patch_size = patch_size
        self.bit_depth = bit_depth
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        with rawpy.imread(self.samples[index]) as raw:
            raw_data = raw.raw_image.astype(np.int16)

            H, W = raw_data.shape
            if self.crop == 'random':
                h_start = random.randint(0, H - 1 - self.patch_size)
                w_start = random.randint(0, W - 1 - self.patch_size)
                raw_data = raw_data[h_start : h_start + self.patch_size, w_start : w_start + self.patch_size]
            elif self.crop == 'center':
                h_start = H // 2 - self.patch_size // 2
                w_start = W // 2 - self.patch_size // 2
                raw_data = raw_data[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
            else:
                raw_data = raw_data
        tensor_raw = raw_data.copy()
        tensor_raw = torch.from_numpy(tensor_raw).unsqueeze(0).to(torch.int16)
        return tensor_raw, 0, 4095


class SonyTestFolder(Dataset):
    def __init__(self, root, split='train', crop='random', patch_size=512, bit_depth=14):
        self.samples = []
        dir_path = os.path.join(root, split)
        for filename in os.listdir(dir_path):
            self.samples.append(os.path.join(dir_path, filename))
        self.crop = crop
        self.patch_size = patch_size
        self.bit_depth = bit_depth
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        with rawpy.imread(self.samples[index]) as raw:
            raw_data = raw.raw_image.astype(np.int16)
            H, W = raw_data.shape
            if self.crop == 'random':
                h_start = random.randint(0, H - 1 - self.patch_size)
                w_start = random.randint(0, W - 1 - self.patch_size)
                raw_data = raw_data[h_start : h_start + self.patch_size, w_start : w_start + self.patch_size]
            elif self.crop == 'center':
                h_start = H // 2 - self.patch_size // 2
                w_start = W // 2 - self.patch_size // 2
                raw_data = raw_data[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
            else:
                raw_data = raw_data
        tensor_raw = raw_data.copy()
        tensor_raw = torch.from_numpy(tensor_raw).unsqueeze(0).to(torch.int16)//4
        return tensor_raw, 0, 4095
    

#nikon train and test 
class RawFolder(Dataset):
    def __init__(self, root, split='train', crop='random', patch_size=512, bit_depth=14):
        self.samples = []
        dir_path = os.path.join(root, split)
        for filename in os.listdir(dir_path):
            self.samples.append(os.path.join(dir_path, filename))
        self.crop = crop
        self.patch_size = patch_size
        self.bit_depth = bit_depth
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        with rawpy.imread(self.samples[index]) as raw:
            raw_data = raw.raw_image.astype(np.int16)

            H, W = raw_data.shape
            if self.crop == 'random':
                h_start = random.randint(0, H - 1 - self.patch_size)
                w_start = random.randint(0, W - 1 - self.patch_size)
                raw_data = raw_data[h_start : h_start + self.patch_size, w_start : w_start + self.patch_size]
            elif self.crop == 'center':
                h_start = H // 2 - self.patch_size // 2
                w_start = W // 2 - self.patch_size // 2
                raw_data = raw_data[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
            else:
                raw_data = raw_data
        tensor_raw = raw_data.copy()
        tensor_raw = torch.from_numpy(tensor_raw).unsqueeze(0).to(torch.int16)
        return tensor_raw.float(), 0, 16383


class SonyFolder(Dataset):
    def __init__(self, root, split='train', crop='random', patch_size=512, bit_depth=14):
        self.samples = []
        dir_path = os.path.join(root, split)
        for filename in os.listdir(dir_path):
            self.samples.append(os.path.join(dir_path, filename))
        self.crop = crop
        self.patch_size = patch_size
        self.bit_depth = bit_depth
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        with np.load(self.samples[index]) as loader:
            raw_data = loader['data']   
            H, W = raw_data.shape
            if self.crop == 'random':
                h_start = random.randint(0, H - 1 - self.patch_size)
                w_start = random.randint(0, W - 1 - self.patch_size)
                raw_data = raw_data[h_start : h_start + self.patch_size, w_start : w_start + self.patch_size]
            elif self.crop == 'center':
                h_start = H // 2 - self.patch_size // 2
                w_start = W // 2 - self.patch_size // 2
                raw_data = raw_data[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
            else:
                raw_data = raw_data
        tensor_raw = raw_data.copy().astype(np.int16)
        tensor_raw = torch.from_numpy(tensor_raw).unsqueeze(0).to(torch.int16)//4 
        return tensor_raw, 0, 4095

#samsung train folder
class SamsungFolder(Dataset):
    def __init__(self, root, split='train', crop='random', patch_size=512, bit_depth=14):
        self.samples = []
        dir_path = os.path.join(root, split)
        for filename in os.listdir(dir_path):
            self.samples.append(os.path.join(dir_path, filename))
        self.crop = crop
        self.patch_size = patch_size
        self.bit_depth = bit_depth
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        with np.load(self.samples[index]) as loader:
            raw_data = loader['data']   

            H, W = raw_data.shape
            if self.crop == 'random':
                h_start = random.randint(0, H - 1 - self.patch_size)
                w_start = random.randint(0, W - 1 - self.patch_size)
                raw_data = raw_data[h_start : h_start + self.patch_size, w_start : w_start + self.patch_size]
            elif self.crop == 'center':
                h_start = H // 2 - self.patch_size // 2
                w_start = W // 2 - self.patch_size // 2
                raw_data = raw_data[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
            else:
                raw_data = raw_data
        tensor_raw = raw_data.copy().astype(np.int16)
        tensor_raw = torch.from_numpy(tensor_raw).unsqueeze(0).to(torch.int16)   
        return tensor_raw, 0, 4095

#olympus train folder
class OlympusFolder(Dataset):
    def __init__(self, root, split='train', crop='random', patch_size=512, bit_depth=14):
        self.samples = []
        dir_path = os.path.join(root, split)
        for filename in os.listdir(dir_path):
            self.samples.append(os.path.join(dir_path, filename))
        self.crop = crop
        self.patch_size = patch_size
        self.bit_depth = bit_depth
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        with np.load(self.samples[index]) as loader:
            raw_data = loader['data']   

            H, W = raw_data.shape
            if self.crop == 'random':
                h_start = random.randint(0, H - 1 - self.patch_size)
                w_start = random.randint(0, W - 1 - self.patch_size)
                raw_data = raw_data[h_start : h_start + self.patch_size, w_start : w_start + self.patch_size]
            elif self.crop == 'center':
                h_start = H // 2 - self.patch_size // 2
                w_start = W // 2 - self.patch_size // 2
                raw_data = raw_data[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
            else:
                raw_data = raw_data
        tensor_raw = raw_data.copy().astype(np.int16)
        tensor_raw = torch.from_numpy(tensor_raw).unsqueeze(0).to(torch.int16)  
        return tensor_raw, 0, 4095


    
if __name__ == '__main__':
    for filename in os.listdir("xxx"):
        path = os.path.join("xxx", filename)
        with rawpy.imread(path) as raw:
            black_level = raw.black_level_per_channel[0]
            white_level = raw.camera_white_level_per_channel[0]
            raw_data = raw.raw_image.astype(np.int16)
            H,W = raw_data.shape
            print(f"shape:H:{H} W:{W}")
            