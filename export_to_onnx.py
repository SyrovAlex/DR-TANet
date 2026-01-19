import torch
import torch.nn as nn
import os
import argparse
from TANet import TANet  # убедитесь, что TANet доступен в PYTHONPATH


def export_to_onnx(checkpoint_path, onnx_path, input_shape=(1, 6, 256, 256), opset=13):
    """
    Экспорт TANet в ONNX с автоматической загрузкой конфигурации из чекпоинта.

    Args:
        checkpoint_path: путь к .pth чекпоинту (должен содержать 'model_state_dict' и 'config')
        onnx_path: путь для сохранения .onnx файла
        input_shape: кортеж (B, C, H, W) — например, (1, 6, 256, 256)
        opset: версия ONNX opset (рекомендуется 13–17)
    """
    device = torch.device('cpu')  # Экспорт на CPU для максимальной совместимости
    dummy_input = torch.randn(input_shape).to(device)

    # Загрузка чекпоинта
    print(f"Загрузка чекпоинта: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'config' not in ckpt:
        raise ValueError("Чекпоинт не содержит ключа 'config'. Убедитесь, что он сохранён через save_checkpoint().")

    config = ckpt['config']
    print(f"Загруженная конфигурация: {config}")

    # Создание модели из config
    model = TANet(
        encoder_arch=config['encoder_arch'],
        local_kernel_size=config['local_kernel_size'],
        stride=config['attn_stride'],
        padding=config['attn_padding'],
        groups=config['attn_groups'],
        drtam=config['drtam'],
        refinement=config['refinement']
    ).to(device)

    # Загрузка весов
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Модель загружена и переведена в eval-режим.")

    # Экспорт в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        },
        verbose=False
    )
    print(f"✅ Модель успешно экспортирована в {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Экспорт TANet в ONNX")
    parser.add_argument("--checkpoint", required=True, help="Путь к .pth чекпоинту (например, best.pth)")
    parser.add_argument("--onnx", required=True, help="Путь для сохранения .onnx файла")
    parser.add_argument("--height", type=int, default=256, help="Высота входного изображения")
    parser.add_argument("--width", type=int, default=256, help="Ширина входного изображения")
    parser.add_argument("--batch", type=int, default=1, help="Размер батча (обычно 1 для инференса)")
    parser.add_argument("--opset", type=int, default=13, choices=[11, 12, 13, 14, 15, 16, 17], help="ONNX opset version")

    args = parser.parse_args()

    input_shape = (args.batch, 6, args.height, args.width)  # 6 каналов: [img_t0, img_t1] по 3 канала

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        input_shape=input_shape,
        opset=args.opset
    )