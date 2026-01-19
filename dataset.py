import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A


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

    def __init__(self, root: str, img_size: int = 256):
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

        target_short = int(img_size * 1.05)  # 268 или 269
        self.common_transform = A.Compose([
            A.SmallestMaxSize(max_size=target_short),
            A.RandomCrop(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.Affine(scale=(0.85, 1.15), translate_percent=0, rotate=0, p=0.4),
        ], additional_targets={"t1_img": "image", "mask_img": "mask"})

        # Аугментации только для t1
        self.t1_only_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.4),
            A.RandomRain(slant_range=(-7, 7), drop_length=8, drop_width=1, brightness_coefficient=0.9, p=0.15),
            # A.RandomFog(fog_coef_range=(0.1, 0.15), alpha_coef=0.001, p=0.05),
            # A.RandomShadow(shadow_intensity_range=(0.3, 0.7), shadow_dimension=4, p=0.15),
            A.GaussNoise(std_range=(0.05, 0.10), p=0.25),
            A.ImageCompression(quality_range=(85, 98), p=0.15),
        ])

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

        augmented = self.common_transform(image=img_t0_raw, t1_img=img_t1_raw, mask=mask_raw)
        aug_t0 = augmented['image']
        aug_t1 = augmented['t1_img']
        aug_mask = augmented['mask']

        aug_t1_final = self.t1_only_transform(image=aug_t1)['image']

        # Normalize images to [-1, 1]
        img_t0 = aug_t0.astype(np.float32).transpose(2, 0, 1) / 128.0 - 1.0
        img_t1 = aug_t1_final.astype(np.float32).transpose(2, 0, 1) / 128.0 - 1.0

        # Binarize mask and convert to long tensor of shape (H, W)
        # 1 - есть изменение, 0 - нет изменения
        mask = (aug_mask > 128).astype(np.int64)  # shape (H, W)

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize and save samples from PCD_Dataset")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset (with t0/, t1/, mask/)")
    parser.add_argument("--output_dir", type=str, default="dataset_preview", help="Directory to save visualized samples")
    parser.add_argument("--img_size", type=int, default=256, help="Image size used in dataset")
    args = parser.parse_args()

    # Создаём датасет
    dataset = PCD_Dataset(root=args.root, img_size=args.img_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset loaded: {len(dataset)} samples. Saving previews to {output_dir}")

    for i in range(len(dataset)):
        try:
            input_tensor, mask_tensor = dataset[i]  # shape: (6, H, W), (H, W)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue

        # Разделяем обратно t0 и t1
        img_t0 = input_tensor[:3].numpy()  # (3, H, W)
        img_t1 = input_tensor[3:].numpy()  # (3, H, W)
        mask = mask_tensor.numpy()         # (H, W), 0/1

        # Обратное преобразование нормализации: [-1, 1] → [0, 255]
        img_t0_vis = ((img_t0 + 1.0) * 128).clip(0, 255).astype(np.uint8)
        img_t1_vis = ((img_t1 + 1.0) * 128).clip(0, 255).astype(np.uint8)

        # Преобразуем в HWC
        img_t0_vis = img_t0_vis.transpose(1, 2, 0)
        img_t1_vis = img_t1_vis.transpose(1, 2, 0)

        # Маска в RGB
        mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

        # Опционально: наложим маску на t1 с прозрачностью
        overlay = cv2.addWeighted(img_t1_vis, 0.7, mask_rgb, 0.3, 0)

        # Собираем всё в одну панель: [t0 | t1 | mask | overlay]
        combined = np.hstack((img_t0_vis, img_t1_vis, mask_rgb, overlay))

        # Сохраняем
        out_path = output_dir / f"sample_{i:04d}.png"
        cv2.imwrite(str(out_path), combined)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")

    print(f"All samples saved to {output_dir}")


if __name__ == "__main__":
    main()
        
        
        
        
        


