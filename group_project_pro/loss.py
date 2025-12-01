import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

class Mono3DLoss(nn.Module):
    """优化版单目3D损失：
    1. 偏航角：周期性损失（解决-π/π边界）
    2. 尺寸：对数损失（更稳定）
    3. 位置：L1损失（鲁棒性）
    """
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="mean")

    def _orientation_loss(self, pred_theta, target_theta):
        """周期性偏航角损失：1 - cos(θ_pred - θ_target)"""
        theta_diff = pred_theta - target_theta
        loss = 1 - torch.cos(theta_diff)
        return loss.mean()

    def _dimension_loss(self, pred_dim, target_dim):
        """尺寸对数损失：L1(log(pred) - log(target))"""
        # 防止log(0)
        pred_dim = torch.clamp(pred_dim, min=1e-6)
        target_dim = torch.clamp(target_dim, min=1e-6)
        log_pred = torch.log(pred_dim)
        log_target = torch.log(target_dim)
        return self.l1_loss(log_pred, log_target)

    def forward(self, pred, target):
        # 拆分参数
        pred_loc = pred[:, :3]    # x,y,z
        pred_dim = pred[:, 3:6]   # l,w,h
        pred_theta = pred[:, 6]   # ry
        
        target_loc = target[:, :3]
        target_dim = target[:, 3:6]
        target_theta = target[:, 6]

        # 计算各部分损失
        loc_loss = self.l1_loss(pred_loc, target_loc)
        dim_loss = self._dimension_loss(pred_dim, target_dim)
        theta_loss = self._orientation_loss(pred_theta, target_theta)

        # 加权总损失
        total_loss = (
            cfg.LOSS_WEIGHTS["location"] * loc_loss +
            cfg.LOSS_WEIGHTS["dimension"] * dim_loss +
            cfg.LOSS_WEIGHTS["orientation"] * theta_loss
        )

        return {
            "total_loss": total_loss,
            "loc_loss": loc_loss.item(),
            "dim_loss": dim_loss.item(),
            "theta_loss": theta_loss.item()
        }