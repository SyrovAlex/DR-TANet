import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


def is_image_file(filename: str) -> bool:
    """Check if a file has a valid image extension."""
    return Path(filename).suffix.lower() in {'.png', '.jpg', '.jpeg'}


# PCD = Pixel-level Change Detection
class PCD_Dataset(Dataset):
    """
    Dataset for binary change detection.
    Expected directory structure:
        root/
            t0/      # pre-change images
            t1/      # post-change images
            mask/    # binary change masks (0 = no change, 1 = change)
    All three directories must contain matching filenames.
    """

    def __init__(self, root: str):
        super().__init__()
        self.root = Path(root)
        self.img_t0_dir = self.root / 't0'
        self.img_t1_dir = self.root / 't1'
        self.mask_dir = self.root / 'mask'

        # Collect valid image basenames from each directory
        t0_names = {f.stem for f in self.img_t0_dir.iterdir() if is_image_file(f.name)}
        t1_names = {f.stem for f in self.img_t1_dir.iterdir() if is_image_file(f.name)}
        mask_names = {f.stem for f in self.mask_dir.iterdir() if is_image_file(f.name)}

        # Keep only files present in all three directories
        self.basenames = sorted(t0_names & t1_names & mask_names)

        if not self.basenames:
            raise ValueError(f"No common image files found in {self.img_t0_dir}, {self.img_t1_dir}, and {self.mask_dir}")

        # Store full filenames with original extensions for robustness
        self.t0_files = [next(self.img_t0_dir.glob(f"{name}.*")) for name in self.basenames]
        self.t1_files = [next(self.img_t1_dir.glob(f"{name}.*")) for name in self.basenames]
        self.mask_files = [next(self.mask_dir.glob(f"{name}.*")) for name in self.basenames]

    def __len__(self) -> int:
        return len(self.basenames)

    def __getitem__(self, index: int):
        fn_t0 = self.t0_files[index]
        fn_t1 = self.t1_files[index]
        fn_mask = self.mask_files[index]

        # Load images
        img_t0_raw = cv2.imread(str(fn_t0), cv2.IMREAD_COLOR)
        img_t1_raw = cv2.imread(str(fn_t1), cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(str(fn_mask), cv2.IMREAD_GRAYSCALE)

        if img_t0_raw is None:
            raise FileNotFoundError(f"Cannot load image: {fn_t0}")
        if img_t1_raw is None:
            raise FileNotFoundError(f"Cannot load image: {fn_t1}")
        if mask_raw is None:
            raise FileNotFoundError(f"Cannot load mask: {fn_mask}")

        # Normalize images to [-1, 1]
        img_t0 = img_t0_raw.astype(np.float32).transpose(2, 0, 1) / 128.0 - 1.0
        img_t1 = img_t1_raw.astype(np.float32).transpose(2, 0, 1) / 128.0 - 1.0

        # Binarize mask and convert to long tensor of shape (H, W)
        # 1 - есть изменение, 0 - нет изменения
        mask = (mask_raw > 128).astype(np.int64)  # shape (H, W)

        # Concatenate inputs along channel dimension: (6, H, W)
        input_tensor = torch.from_numpy(np.concatenate((img_t0, img_t1), axis=0))
        mask_tensor = torch.from_numpy(mask).long()  # shape (H, W)

        return input_tensor, mask_tensor

    def __repr__(self) -> str:
        return f"PCD Dataset | Root: {self.root} | Samples: {len(self)}"

    # Optional: keep only if actively used for debugging
    def get_random_sample(self):
        idx = np.random.randint(len(self))
        return self[idx]

        
        
        
        
        


