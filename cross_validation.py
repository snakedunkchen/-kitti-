"""
交叉验证脚本：实现K折交叉验证以评估模型泛化能力
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
from config import cfg
from dataset import KITTIDataset, build_transforms
from model import build_model
from loss import Mono3DLoss
from metrics import KITTIMetrics
from utils import logger
import matplotlib.pyplot as plt
import json

class CrossValidator:
    def __init__(self, n_splits=5, model_type='attention'):
        self.n_splits = n_splits
        self.model_type = model_type
        self.fold_results = []
        
    def train_one_epoch(self, model, train_loader, criterion, optimizer):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        
        for images, targets, _ in train_loader:
            images = images.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            # 前向传播
            preds = model(images)
            loss_dict = criterion(preds, targets)
            loss = loss_dict["total_loss"]
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        
        return total_loss / len(train_loader.dataset)
    
    @torch.no_grad()
    def validate(self, model, val_loader, criterion):
        """验证"""
        model.eval()
        total_loss = 0.0
        metrics = KITTIMetrics()
        
        for images, targets, _ in val_loader:
            images = images.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            preds = model(images)
            loss_dict = criterion(preds, targets)
            loss = loss_dict["total_loss"]
            
            total_loss += loss.item() * images.size(0)
            metrics.update(preds, targets)
        
        avg_loss = total_loss / len(val_loader.dataset)
        eval_metrics = metrics.compute()
        
        return avg_loss, eval_metrics
    
    def run_fold(self, fold, train_indices, val_indices, dataset):
        """运行单个折"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{self.n_splits}")
        logger.info(f"{'='*50}")
        
        # 创建数据加载器
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            dataset,
            batch_size=cfg.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=cfg.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # 构建模型
        original_type = cfg.BACKBONE_TYPE
        cfg.BACKBONE_TYPE = self.model_type
        model = build_model()
        cfg.BACKBONE_TYPE = original_type
        
        # 损失函数和优化器
        criterion = Mono3DLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
        
        # 训练
        best_val_loss = float('inf')
        best_metrics = None
        patience_counter = 0
        
        num_epochs = min(20, cfg.EPOCHS)  # 交叉验证使用较少的epoch
        
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_metrics = self.validate(model, val_loader, criterion)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"AP@0.5: {val_metrics.get('AP_0.5', 0):.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    logger.info(f"早停触发于 epoch {epoch+1}")
                    break
        
        fold_result = {
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'metrics': best_metrics,
            'train_size': len(train_indices),
            'val_size': len(val_indices)
        }
        
        self.fold_results.append(fold_result)
        
        return fold_result
    
    def run(self):
        """运行完整的交叉验证"""
        logger.info(f"开始 {self.n_splits}-折交叉验证...")
        logger.info(f"模型类型: {self.model_type}")
        
        # 加载数据集
        dataset = KITTIDataset(
            root=cfg.DATA_ROOT,
            split="training",
            transform=build_transforms(train=True)
        )
        
        # K折分割
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=cfg.SEED)
        
        # 运行每一折
        for fold, (train_indices, val_indices) in enumerate(kfold.split(range(len(dataset)))):
            self.run_fold(fold, train_indices, val_indices, dataset)
        
        # 计算平均结果
        self.compute_average_results()
        
        # 可视化结果
        self.visualize_results()
        
        # 保存结果
        self.save_results()
    
    def compute_average_results(self):
        """计算平均结果"""
        logger.info(f"\n{'='*50}")
        logger.info("交叉验证汇总结果")
        logger.info(f"{'='*50}")
        
        # 收集所有指标
        all_losses = [r['best_val_loss'] for r in self.fold_results]
        
        # 收集所有AP指标
        metric_keys = self.fold_results[0]['metrics'].keys()
        avg_metrics = {}
        
        for key in metric_keys:
            values = [r['metrics'][key] for r in self.fold_results]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        # 打印结果
        logger.info(f"\n平均验证损失: {np.mean(all_losses):.4f} ± {np.std(all_losses):.4f}")
        logger.info("\n平均评估指标:")
        for key, stats in avg_metrics.items():
            logger.info(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 打印每一折的详细结果
        logger.info(f"\n{'Fold':<8} {'Val Loss':<12} {'AP@0.5':<12} {'AP@0.7':<12}")
        logger.info("-" * 50)
        for result in self.fold_results:
            logger.info(f"{result['fold']:<8} "
                       f"{result['best_val_loss']:<12.4f} "
                       f"{result['metrics'].get('AP_0.5', 0):<12.4f} "
                       f"{result['metrics'].get('AP_0.7', 0):<12.4f}")
        
        self.avg_metrics = avg_metrics
        self.avg_loss = np.mean(all_losses)
        self.std_loss = np.std(all_losses)
    
    def visualize_results(self):
        """可视化交叉验证结果"""
        vis_dir = os.path.join(cfg.VIS_DIR, 'cross_validation')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. 每折的验证损失
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        folds = [r['fold'] for r in self.fold_results]
        losses = [r['best_val_loss'] for r in self.fold_results]
        ap_05 = [r['metrics'].get('AP_0.5', 0) for r in self.fold_results]
        
        axes[0].bar(folds, losses, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0].axhline(y=self.avg_loss, color='r', linestyle='--', 
                       label=f'Mean: {self.avg_loss:.4f}')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title(f'{self.n_splits}-Fold Cross-Validation - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].bar(folds, ap_05, alpha=0.7, edgecolor='black', color='lightcoral')
        axes[1].axhline(y=self.avg_metrics['AP_0.5']['mean'], color='r', linestyle='--',
                       label=f"Mean: {self.avg_metrics['AP_0.5']['mean']:.4f}")
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('AP@0.5')
        axes[1].set_title(f'{self.n_splits}-Fold Cross-Validation - AP@0.5')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'cv_{self.model_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 箱线图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].boxplot([losses], labels=[self.model_type])
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Validation Loss Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot([ap_05], labels=[self.model_type])
        axes[1].set_ylabel('AP@0.5')
        axes[1].set_title('AP@0.5 Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'cv_boxplot_{self.model_type}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化结果保存到: {vis_dir}")
    
    def save_results(self):
        """保存交叉验证结果"""
        results = {
            'model_type': self.model_type,
            'n_splits': self.n_splits,
            'fold_results': self.fold_results,
            'average_loss': float(self.avg_loss),
            'std_loss': float(self.std_loss),
            'average_metrics': {k: {'mean': float(v['mean']), 'std': float(v['std'])} 
                               for k, v in self.avg_metrics.items()}
        }
        
        results_file = os.path.join(cfg.LOG_DIR, f'cv_results_{self.model_type}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"交叉验证结果保存到: {results_file}")

def main():
    """主函数"""
    # 对多个模型进行交叉验证
    models = ['simple', 'resnet', 'attention']
    
    all_results = {}
    
    for model_type in models:
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"交叉验证模型: {model_type}")
        logger.info(f"{'#'*60}\n")
        
        try:
            validator = CrossValidator(n_splits=5, model_type=model_type)
            validator.run()
            all_results[model_type] = {
                'avg_loss': validator.avg_loss,
                'avg_metrics': validator.avg_metrics
            }
        except Exception as e:
            logger.error(f"交叉验证 {model_type} 时出错: {str(e)}")
            continue
    
    # 比较所有模型
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("所有模型交叉验证对比")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"{'Model':<15} {'Avg Loss':<15} {'AP@0.5':<15} {'AP@0.7':<15}")
        logger.info("-" * 60)
        for model, results in all_results.items():
            logger.info(f"{model:<15} "
                       f"{results['avg_loss']:<15.4f} "
                       f"{results['avg_metrics']['AP_0.5']['mean']:<15.4f} "
                       f"{results['avg_metrics']['AP_0.7']['mean']:<15.4f}")
    
    logger.info("\n交叉验证完成！")

if __name__ == "__main__":
    main()
