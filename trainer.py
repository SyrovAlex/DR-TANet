import os
import csv
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from TANet import TANet
from dataset import PCD_Dataset
from loss import ChangeDetectionLoss


class ChangeDetectionTrainer:
    """Продвинутый тренер для DR-TANet / TANet с AMP, gradient clipping, TensorBoard и ранней остановкой"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # === DataLoader ===
        self.train_dataset = PCD_Dataset(config['train_dir'])
        self.val_dataset = PCD_Dataset(config['val_dir'])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.get('val_batch_size', 1),
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        # === Модель ===
        self.model = TANet(
            encoder_arch=config['encoder_arch'],
            local_kernel_size=config['local_kernel_size'],
            attn_stride=config['attn_stride'],
            attn_padding=config['attn_padding'],
            attn_groups=config['attn_groups'],
            drtam=config['drtam'],
            refinement=config['refinement']
        ).to(self.device)

        if config['multi_gpu'] and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            print(f"Используется {torch.cuda.device_count()} GPU")

        # === Loss ===
        pos_weight = config.get('pos_weight', 1.0)
        weight = torch.tensor([1.0, pos_weight]).to(self.device)
        self.criterion = ChangeDetectionLoss(
            alpha=config.get('loss_alpha', 0.5),
            gamma=config.get('focal_gamma', 2.0),
            use_focal=True,
            use_dice=True
        )

        # === Optimizer ===
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 0.0)
        )

        # === Scheduler ===
        total_steps = config['epochs'] * len(self.train_loader)
        scheduler_type = config.get('scheduler', 'linear')
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['epochs']
            )
        elif scheduler_type == 'step':
            step_size = config.get('step_size', 30)
            gamma = config.get('gamma', 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps)
            )
        else:
            raise ValueError(f"Неизвестный scheduler: {scheduler_type}")

        # === Ранняя остановка ===
        self.best_metric = -float('inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 10)

        # === Логирование ===
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.loss_log_path = os.path.join(self.checkpoint_dir, 'loss.csv')

        print(f"Устройство: {self.device}")
        print(f"AMP: {'включён' if self.use_amp else 'выключен'}")
        print(f"Gradient clipping: {config.get('grad_clip', 0.0)}")
        print(f"Scheduler: {scheduler_type}")
        print(f"Модель: {'DR-TANet' if config['drtam'] else 'TANet'}, Encoder: {config['encoder_arch']}")

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        with open(self.loss_log_path, 'a', newline='') as f_loss:
            loss_writer = csv.writer(f_loss)
            if self.epoch == 0:
                loss_writer.writerow(['step', 'loss'])

            progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}', leave=False)
            for inputs, masks in progress_bar:
                inputs = inputs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                with autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                grad_clip = self.config.get('grad_clip', 0.0)
                if grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_item = loss.item()
                epoch_loss += loss_item
                self.step += 1
                loss_writer.writerow([self.step, loss_item])

                progress_bar.set_postfix({'loss': f'{loss_item:.4f}'})

        avg_loss = epoch_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def compute_validation_metrics(self):
        self.model.eval()

        all_f1, all_iou = [], []
        global_f1 = BinaryF1Score().to(self.device)
        global_iou = BinaryJaccardIndex().to(self.device)

        for inputs, masks in tqdm(self.val_loader, desc="Validating", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1)

            global_f1.update(preds, masks)
            global_iou.update(preds, masks)

            for i in range(preds.size(0)):
                p, t = preds[i].unsqueeze(0), masks[i].unsqueeze(0)
                f1_img = BinaryF1Score().to(self.device)
                iou_img = BinaryJaccardIndex().to(self.device)
                f1_img.update(p, t)
                iou_img.update(p, t)
                all_f1.append(f1_img.compute().item())
                all_iou.append(iou_img.compute().item())

        return {
            'mean_f1': np.mean(all_f1) if all_f1 else 0.0,
            'mean_iou': np.mean(all_iou) if all_iou else 0.0,
            'global_f1': global_f1.compute().item(),
            'global_iou': global_iou.compute().item()
        }

    @torch.no_grad()
    def visualize_samples(self, num_samples=4):
        self.model.eval()
        sample_dir = os.path.join(self.checkpoint_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        for i in range(min(num_samples, len(self.val_dataset))):
            input_tensor, mask_gt = self.val_dataset[i]
            input_batch = input_tensor.unsqueeze(0).to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                output = self.model(input_batch)

            pred_mask = torch.argmax(output[0], dim=0).cpu().numpy()
            gt_mask = mask_gt.numpy()

            img_t0 = np.transpose((input_tensor[0:3] + 1) * 128, (1, 2, 0)).astype(np.uint8)
            img_t1 = np.transpose((input_tensor[3:6] + 1) * 128, (1, 2, 0)).astype(np.uint8)
            pred_rgb = np.stack([pred_mask * 255] * 3, axis=-1).astype(np.uint8)
            gt_rgb = np.stack([gt_mask * 255] * 3, axis=-1).astype(np.uint8)

            h, w = img_t0.shape[:2]
            canvas = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
            canvas[0:h, 0:w] = img_t0
            canvas[0:h, w:2*w] = img_t1
            canvas[h:2*h, 0:w] = gt_rgb
            canvas[h:2*h, w:2*w] = pred_rgb

            cv2.imwrite(
                os.path.join(sample_dir, f'sample_epoch{self.epoch:03d}_{i}.png'),
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            )

    def log_and_save_metrics(self, metrics):
        # CSV
        path = os.path.join(self.checkpoint_dir, 'val_metrics.csv')
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'mean_f1', 'mean_iou', 'global_f1', 'global_iou'])
            writer.writerow([
                self.epoch + 1,
                metrics['mean_f1'],
                metrics['mean_iou'],
                metrics['global_f1'],
                metrics['global_iou']
            ])

        # TensorBoard
        self.writer.add_scalar('Val/Mean_F1', metrics['mean_f1'], self.epoch)
        self.writer.add_scalar('Val/Mean_IoU', metrics['mean_iou'], self.epoch)
        self.writer.add_scalar('Val/Global_F1', metrics['global_f1'], self.epoch)
        self.writer.add_scalar('Val/Global_IoU', metrics['global_iou'], self.epoch)

        print(
            f"[Val] Epoch {self.epoch + 1} | "
            f"Mean IoU: {metrics['mean_iou']:.4f}, Mean F1: {metrics['mean_f1']:.4f}"
        )

    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'config': self.config
        }
        torch.save(state, os.path.join(self.checkpoint_dir, 'last.pth'))
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best.pth'))

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch'] + 1
            self.best_metric = checkpoint.get('best_metric', -float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            print(f"Загружен чекпоинт до эпохи {checkpoint['epoch']}")

    def train(self, resume=False):
        self.step = 0
        if resume:
            self.load_checkpoint(os.path.join(self.checkpoint_dir, 'last.pth'))

        for self.epoch in range(self.epoch, self.config['epochs']):
            train_loss = self.train_epoch()
            print(f"[Train] Epoch {self.epoch + 1} | Loss: {train_loss:.4f}")

            # Валидация раз в N эпох
            val_interval = self.config.get('val_interval', 1)
            if (self.epoch + 1) % val_interval == 0:
                val_metrics = self.compute_validation_metrics()
                self.log_and_save_metrics(val_metrics)

                # Визуализация
                if (self.epoch + 1) % self.config.get('val_vis_interval', 5) == 0:
                    self.visualize_samples()

                # Ранняя остановка (по mean IoU)
                current_metric = val_metrics['mean_iou']
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                    print(f"🏆 Новый рекорд! Mean IoU: {current_metric:.4f}")
                else:
                    self.patience_counter += 1

                # Сохранение последнего чекпоинта
                self.save_checkpoint()

                if self.patience_counter >= self.patience:
                    print(f"⏹️ Ранняя остановка на эпохе {self.epoch + 1}")
                    break

            # Шаг scheduler'а (если не linear по шагам)
            if self.config.get('scheduler', 'linear') != 'linear':
                self.scheduler.step()

        self.writer.close()
        print("✅ Обучение завершено.")








