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
    # Загрузка фонового изображения
    bg_img_orig = cv2.imread(args.background)
    if bg_img_orig is None:
        raise FileNotFoundError(f"Не найден фон: {args.background}")

    # Открытие входного видео
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {args.input_video}")

    # Параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Настройка выходного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # можно 'XVID', 'H264' и т.д.
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise RuntimeError(f"Не удалось создать выходное видео: {args.output_video}")

    # Загрузка ONNX модели
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if not args.cpu else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(args.onnx_model, providers=providers)
    print(f"ONNX модель загружена. Используемые провайдеры: {ort_session.get_providers()}")

    img_size = args.img_size
    threshold = args.threshold

    with tqdm(total=total_frames, desc="Обработка видео") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_img_orig = frame.copy()

            # Letterbox
            bg_img, bg_params = letterbox(bg_img_orig, new_shape=(img_size, img_size))
            curr_img, curr_params = letterbox(curr_img_orig, new_shape=(img_size, img_size))

            # Нормализация и подготовка входа
            bg_norm = normalize_image(bg_img)
            curr_norm = normalize_image(curr_img)
            input_np = np.concatenate([bg_norm, curr_norm], axis=2)  # [H, W, 6]
            input_np = np.transpose(input_np, (2, 0, 1))             # [6, H, W]
            input_batch = np.expand_dims(input_np, axis=0)           # [1, 6, H, W]

            # Инференс
            outputs = ort_session.run(None, {'input': input_batch})
            logits = outputs[0]  # [1, 2, H, W]

            # Вероятности и порог
            probs = np.exp(logits[0])
            probs = probs / np.sum(probs, axis=0, keepdims=True)
            change_prob = probs[1]
            pred_mask_lb = (change_prob >= threshold).astype(np.uint8)

            # Обратное преобразование маски
            orig_shape = curr_img_orig.shape[:2]
            pred_mask_orig = mask_postprocess(pred_mask_lb, orig_shape, curr_params)

            # Наложение маски
            overlay = curr_img_orig.copy()
            overlay[pred_mask_orig == 1] = [0, 0, 255]  # красный
            alpha = 0.5
            overlay = cv2.addWeighted(curr_img_orig, 1 - alpha, overlay, alpha, 0)

            # Запись кадра
            out.write(overlay)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Готово! Результат сохранён в: {args.output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Инференс ONNX-модели на видео с наложением маски изменений"
    )
    parser.add_argument("--background", required=True, help="Путь к фоновому изображению (референсу)")
    parser.add_argument("--input_video", required=True, help="Путь к входному видео")
    parser.add_argument("--onnx_model", required=True, help="Путь к .onnx файлу модели")
    parser.add_argument("--output_video", default="output.mp4", help="Путь к выходному видео")
    parser.add_argument("--cpu", action="store_true", help="Использовать CPU вместо GPU")
    parser.add_argument("--img_size", type=int, default=640, help="Целевой размер изображения после letterbox")
    parser.add_argument("--threshold", type=float, default=0.85, help="Порог вероятности для бинаризации маски (0.0–1.0)")

    args = parser.parse_args()
    main(args)