import os
import cv2
import numpy as np
import argparse
import onnxruntime as ort
from tqdm import tqdm


def normalize_image(img):
    """BGR [0,255] → RGB → [-1, 1]"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2.0  # [-1, 1]
    return img


def letterbox(img, new_shape=(512, 512), color=(0, 0, 0)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, (left, top))


def mask_postprocess(pred_mask, orig_shape, letterbox_params):
    r, (pad_left, pad_top) = letterbox_params
    h_unpad = int(round(orig_shape[0] * r))
    w_unpad = int(round(orig_shape[1] * r))

    y1 = pad_top
    y2 = pad_top + h_unpad
    x1 = pad_left
    x2 = pad_left + w_unpad

    active = pred_mask[y1:y2, x1:x2]
    mask_resized = cv2.resize(active, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized


def main(args):
    bg_img_orig = cv2.imread(args.background)
    if bg_img_orig is None:
        raise FileNotFoundError(f"Не найден фон: {args.background}")

    current_files = sorted([
        f for f in os.listdir(args.current_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not current_files:
        raise ValueError(f"Папка {args.current_dir} не содержит изображений.")

    os.makedirs(args.output_dir, exist_ok=True)

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if not args.cpu else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(args.onnx_model, providers=providers)
    print(f"ONNX модель загружена. Используемые провайдеры: {ort_session.get_providers()}")

    img_size = args.img_size
    threshold = args.threshold

    with tqdm(total=len(current_files), desc="Обработка изображений") as pbar:
        for filename in current_files:
            curr_path = os.path.join(args.current_dir, filename)
            curr_img_orig = cv2.imread(curr_path)
            if curr_img_orig is None:
                print(f"⚠️ Пропущено: не удалось загрузить {curr_path}")
                continue

            bg_img, bg_params = letterbox(bg_img_orig, new_shape=(img_size, img_size))
            curr_img, curr_params = letterbox(curr_img_orig, new_shape=(img_size, img_size))

            bg_norm = normalize_image(bg_img)
            curr_norm = normalize_image(curr_img)
            input_np = np.concatenate([bg_norm, curr_norm], axis=2)
            input_np = np.transpose(input_np, (2, 0, 1))
            input_batch = np.expand_dims(input_np, axis=0)

            outputs = ort_session.run(None, {'input': input_batch})
            logits = outputs[0]  # [1, 2, H, W]

            # --- Новое: используем вероятности и порог ---
            # Применяем softmax по каналам (axis=1)
            probs = np.exp(logits[0])  # [2, H, W]
            probs = probs / np.sum(probs, axis=0, keepdims=True)  # нормализация
            change_prob = probs[1]  # вероятность класса "изменение"

            # Применяем порог
            pred_mask_lb = (change_prob >= threshold).astype(np.uint8)  # [H, W]

            # Обратное преобразование маски
            orig_shape = curr_img_orig.shape[:2]
            pred_mask_orig = mask_postprocess(pred_mask_lb, orig_shape, curr_params)

            # Наложение маски
            overlay = curr_img_orig.copy()
            overlay[pred_mask_orig == 1] = [0, 0, 255]  # красный

            # Полупрозрачный оверлей
            alpha = 0.5
            overlay = cv2.addWeighted(curr_img_orig, 1 - alpha, overlay, alpha, 0)

            # Сохранение
            output_path = os.path.join(args.output_dir, filename)
            cv2.imwrite(output_path, overlay)

            pbar.update(1)

    print(f"✅ Готово! Результаты сохранены в: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Инференс ONNX-модели с порогом по вероятности и наложением маски на исходные изображения"
    )
    parser.add_argument("--background", required=True, help="Путь к одному фоновому изображению (референсу)")
    parser.add_argument("--current_dir", required=True, help="Папка с текущими (изменёнными) изображениями")
    parser.add_argument("--onnx_model", required=True, help="Путь к .onnx файлу модели")
    parser.add_argument("--output_dir", default="onnx_composites", help="Папка для сохранения результатов")
    parser.add_argument("--cpu", action="store_true", help="Использовать CPU вместо GPU")
    parser.add_argument("--img_size", type=int, default=512, help="Целевой размер изображения после letterbox")
    parser.add_argument("--threshold", type=float, default=0.95, help="Порог вероятности для бинаризации маски (от 0.0 до 1.0)")

    args = parser.parse_args()
    main(args)