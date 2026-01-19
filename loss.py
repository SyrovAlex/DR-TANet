import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeDetectionLoss(nn.Module):
    def __init__(
            self,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            dice_weight: float = 0.5,
            use_focal: bool = True,
            use_dice: bool = True,
    ):
        """
        Комбинированный лосс для бинарной детекции изменений.

        Args:
            focal_alpha: вес редкого класса в Focal Loss (обычно 0.25–0.75)
            focal_gamma: фокусирующий параметр (обычно 2.0)
            dice_weight: вес Dice Loss в общей сумме (от 0 до 1)
            use_focal: использовать Focal Loss
            use_dice: использовать Dice Loss
        """
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        self.use_dice = use_dice

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss для 2-классовой сегментации.

        Args:
            pred: logits, shape (N, 2, H, W)
            target: ground truth, shape (N, H, W), values in {0, 1}
        Returns:
            scalar loss
        """
        # Лог-вероятности
        log_prob = F.log_softmax(pred, dim=1)  # (N, 2, H, W)
        prob = torch.exp(log_prob)  # (N, 2, H, W)

        # One-hot целевой тензор
        target_one_hot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()  # (N, 2, H, W)

        # Веса для балансировки классов: alpha для класса 1, (1-alpha) для класса 0
        alpha_weight = self.focal_alpha * target_one_hot + (1 - self.focal_alpha) * (1 - target_one_hot)

        # Focal модуляция: (1 - p_t)^gamma
        focal_modulation = (1 - prob) ** self.focal_gamma

        # Итоговый loss: -alpha * (1 - p_t)^gamma * log(p_t)
        focal_loss = -alpha_weight * focal_modulation * log_prob
        return focal_loss.mean()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Soft Dice Loss для класса "изменение" (канал 1).

        Args:
            pred: logits, shape (N, 2, H, W)
            target: ground truth, shape (N, H, W)
        Returns:
            scalar loss
        """
        # Вероятность класса "изменение"
        prob_change = torch.softmax(pred, dim=1)[:, 1, :, :]  # (N, H, W)
        target_float = target.float()  # (N, H, W)

        # Выравнивание размеров
        prob_flat = prob_change.contiguous().view(-1)
        target_flat = target_float.contiguous().view(-1)

        intersection = (prob_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (prob_flat.sum() + target_flat.sum() + smooth)
        return 1.0 - dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        if self.use_focal:
            focal = self.focal_loss(pred, target)
            total_loss += (1.0 - self.dice_weight) * focal

        if self.use_dice:
            dice = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice

        return total_loss