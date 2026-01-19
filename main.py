import os
import argparse
from trainer import ChangeDetectionTrainer  # Предполагается, что ваш тренер находится в trainer.py


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение DR-TANet / TANet для обнаружения изменений")

    # -----------------------------
    # Пути к данным
    # -----------------------------
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Путь к тренировочным данным (должен содержать подпапки t0/, t1/, mask/)')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Путь к валидационным данным')
    parser.add_argument('--checkpoint-dir', type=str, default='./runs/dr_tanet',
                        help='Директория для сохранения чекпоинтов, логов и визуализаций')

    # -----------------------------
    # Архитектура модели
    # -----------------------------
    parser.add_argument('--encoder-arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Архитектура энкодера')
    parser.add_argument('--drtam', action='store_true',
                        help='Использовать DR-TANet (иначе — стандартный TANet)')
    parser.add_argument('--refinement', action='store_true',
                        help='Добавить модуль уточнения (refinement module)')

    # Параметры TANet / DR-TANet
    parser.add_argument('--local-kernel-size', type=int, default=1,
                        help='Размер ядра локального внимания (по умолчанию: 1)')
    parser.add_argument('--attn-stride', type=int, default=1,
                        help='Stride для модуля внимания')
    parser.add_argument('--attn-padding', type=int, default=0,
                        help='Padding для модуля внимания')
    parser.add_argument('--attn-groups', type=int, default=4,
                        help='Количество групп в attention')

    # -----------------------------
    # Гиперпараметры обучения
    # -----------------------------
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--batch-size', type=int, default=8, help='Размер батча для обучения')
    parser.add_argument('--val-batch-size', type=int, default=4, help='Размер батча для валидации')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Базовый learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 регуляризация')
    parser.add_argument('--pos-weight', type=float, default=2.0,
                        help='Вес положительного класса (изменение) в Focal Loss')

    # Параметры Combined Loss
    parser.add_argument('--loss-alpha', type=float, default=0.5,
                        help='Вес Focal Loss в комбинированной функции потерь (1 - alpha — вес Dice)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Gamma для Focal Loss')

    # -----------------------------
    # AMP и оптимизация
    # -----------------------------
    parser.add_argument('--use-amp', action='store_true',
                        help='Использовать Automatic Mixed Precision (fp16)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Максимальная норма градиента для clipping (0 — отключено)')

    # -----------------------------
    # Learning Rate Scheduler
    # -----------------------------
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine', 'step'],
                        help='Тип scheduler\'а')
    parser.add_argument('--step-size', type=int, default=30,
                        help='Шаг для StepLR (актуально при --scheduler step)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Коэффициент снижения LR для StepLR')

    # -----------------------------
    # Валидация и ранняя остановка
    # -----------------------------
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Выполнять валидацию каждые N эпох')
    parser.add_argument('--val-vis-interval', type=int, default=10,
                        help='Сохранять визуализации каждые N эпох')
    parser.add_argument('--patience', type=int, default=15,
                        help='Количество эпох без улучшения до ранней остановки')

    # -----------------------------
    # Оборудование
    # -----------------------------
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Устройство для обучения')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Использовать все доступные GPU (DataParallel)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Количество потоков для загрузки данных')

    # -----------------------------
    # Продолжение обучения
    # -----------------------------
    parser.add_argument('--resume', action='store_true',
                        help='Продолжить обучение с последнего чекпоинта')

    return parser.parse_args()


def main():
    args = parse_args()

    # Формируем конфигурацию
    config = {
        # Пути
        'train_dir': "../dataset1/train",
        'val_dir': "../dataset1/test",
        'checkpoint_dir': "checkpoint",

        # Модель
        'encoder_arch': "resnet18",
        'drtam': False,
        'refinement': False,
        'local_kernel_size': 3,
        'attn_stride': 1,
        'attn_padding': 1,
        'attn_groups': 4,
        'multi_gpu': True,
        "resume": False,

        # Обучение
        'epochs': 50,
        'batch_size': 24,
        'val_batch_size': 1,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'pos_weight': 2.0,

        # Loss
        'focal_alpha': 0.5,
        'focal_gamma': 2.0,

        # AMP и градиенты
        'use_amp': True,
        'grad_clip': 1.0,

        # Валидация
        'val_interval': 1,
        'val_vis_interval': 1,
        'patience': 15,

        # Устройство
        'device': "cuda",
        'num_workers': 1
    }

    # Создаём директорию чекпоинтов
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Запуск обучения
    trainer = ChangeDetectionTrainer(config)
    trainer.train(resume=args.resume)


if __name__ == '__main__':
    main()