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