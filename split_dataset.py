import os
import shutil
import argparse
import random
from pathlib import Path


def split_dataset(t0_dir, t1_dir, mask_dir, output_dir, val_ratio=0.2, seed=42):
    t0_dir = Path(t0_dir)
    t1_dir = Path(t1_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    # Поддерживаемые расширения
    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # Получаем список файлов в каждой папке
    def get_image_files(folder):
        return {f.name for f in folder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS}

    t0_files = get_image_files(t0_dir)
    t1_files = get_image_files(t1_dir)
    mask_files = get_image_files(mask_dir)

    # Находим пересечение имён
    common_files = t0_files & t1_files & mask_files
    if not common_files:
        raise ValueError("Нет общих файлов во всех трёх папках!")

    print(f"Найдено {len(common_files)} совпадающих троек (t0, t1, mask).")

    # Сортируем для воспроизводимости
    common_files = sorted(common_files)
    random.seed(seed)
    random.shuffle(common_files)

    # Разделение
    n_val = int(len(common_files) * val_ratio)
    val_files = set(common_files[:n_val])
    train_files = set(common_files[n_val:])

    print(f"Разделение: {len(train_files)} train, {len(val_files)} val")

    # Создание структуры выходных папок
    for split in ['train', 'val']:
        for subdir in ['t0', 't1', 'mask']:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    # Копирование файлов
    def copy_files(file_set, split_name):
        for fname in file_set:
            shutil.copy(t0_dir / fname, output_dir / split_name / 't0' / fname)
            shutil.copy(t1_dir / fname, output_dir / split_name / 't1' / fname)
            shutil.copy(mask_dir / fname, output_dir / split_name / 'mask' / fname)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    print(f"✅ Данные успешно разделены и сохранены в: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разделение датасета на train/val")
    parser.add_argument("--t0", required=True, help="Папка с фоновыми изображениями (t0)")
    parser.add_argument("--t1", required=True, help="Папка с текущими изображениями (t1)")
    parser.add_argument("--mask", required=True, help="Папка с масками")
    parser.add_argument("--output", required=True, help="Выходная папка для train/val")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Доля валидационной выборки (по умолчанию 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Сид для воспроизводимости")

    args = parser.parse_args()

    split_dataset(
        t0_dir=args.t0,
        t1_dir=args.t1,
        mask_dir=args.mask,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed
    )