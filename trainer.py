import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import os
from tqdm import tqdm
from config import cfg
from loss import Mono3DLoss
from metrics import KITTIMetrics
from utils import logger, save_checkpoint, visualize_3d_box, load_checkpoint

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 损失函数
        self.criterion = Mono3DLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=cfg.LR_SCHEDULER_PATIENCE,
            factor=cfg.LR_SCHEDULER_FACTOR,
        )
        
        # 评估指标
        self.metrics = KITTIMetrics()
        
        # 早停相关
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def train_one_epoch(self, epoch):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for batch_idx, (images, targets, _) in enumerate(pbar):
            # 数据移到设备
            images = images.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            # 前向传播
            preds = self.model(images)
            loss_dict = self.criterion(preds, targets)
            loss = loss_dict["total_loss"]
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.GRAD_CLIP_NORM)
            
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item() * images.size(0)
            
            # 进度条显示
            pbar.set_postfix({
                "loss": loss.item(),
                "loc_loss": loss_dict["loc_loss"],
                "dim_loss": loss_dict["dim_loss"],
                "theta_loss": loss_dict["theta_loss"],
                "lr": self.optimizer.param_groups[0]["lr"]
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader.dataset)
        logger.info(f"Train Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        """验证单个epoch"""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for batch_idx, (images, targets, img_names) in enumerate(pbar):
            images = images.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            # 前向传播
            preds = self.model(images)
            loss_dict = self.criterion(preds, targets)
            loss = loss_dict["total_loss"]
            
            # 累计损失
            total_loss += loss.item() * images.size(0)
            
            # 更新评估指标
            self.metrics.update(preds, targets)
            
            # 进度条显示
            pbar.set_postfix({"val_loss": loss.item()})
            
            # 可视化（每N轮）
            if epoch % cfg.VIS_EVERY_EPOCH == 0 and batch_idx == 0:
                # 简化：取第一张图像可视化
                img = images[0].detach().cpu().numpy().transpose(1,2,0)
                img = (img * cfg.STD + cfg.MEAN).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # 投影3D框（简化：使用默认内参）
                calib_matrix = np.array([[721.5377, 0, 609.5593],
                                         [0, 721.5377, 172.854],
                                         [0, 0, 1]])
                pred_box = preds[0].detach().cpu().numpy()
                img = visualize_3d_box(img, pred_box, calib_matrix)
                
                # 保存可视化结果
                vis_path = os.path.join(cfg.VIS_DIR, f"epoch_{epoch+1}_vis.png")
                cv2.imwrite(vis_path, img)
                logger.info(f"可视化结果保存到: {vis_path}")
        
        # 计算平均损失和评估指标
        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = self.metrics.compute()
        
        # 打印评估结果
        logger.info(f"Val Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        for k, v in metrics.items():
            logger.info(f"Val {k}: {v:.4f}")
        
        # 学习率调度
        self.scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.early_stop_counter = 0
            save_checkpoint(self.model, self.optimizer, epoch, avg_loss, metrics)
        else:
            self.early_stop_counter += 1
            logger.warning(f"早停计数器: {self.early_stop_counter}/{cfg.EARLY_STOP_PATIENCE}")
        
        return avg_loss, metrics

    def train(self):
        """完整训练流程"""
        logger.info("开始训练单目3D目标检测模型...")
        logger.info(f"使用设备: {cfg.DEVICE}")
        logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        
        # 加载预训练模型
        start_epoch, _ = load_checkpoint(self.model, self.optimizer) if cfg.LOAD_CHECKPOINT else (0, 0)
        
        for epoch in range(start_epoch, cfg.EPOCHS):
            # 训练
            train_loss = self.train_one_epoch(epoch)
            
            # 验证
            val_loss, metrics = self.validate(epoch)
            
            # 早停判断
            if self.early_stop_counter >= cfg.EARLY_STOP_PATIENCE:
                logger.info(f"早停触发，停止训练（最佳验证损失: {self.best_val_loss:.4f}）")
                break
        
        logger.info("训练完成！")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")