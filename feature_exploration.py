"""
特征探索脚本：探索不同特征表示对模型性能的影响
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import cfg
from dataset import build_dataloaders
from model import build_model
from trainer import Trainer
from utils import logger
import json

class FeatureExplorer:
    """探索不同特征表示的影响"""
    
    def __init__(self):
        self.results = {}
    
    def explore_feature_representation(self, feature_type):
        """探索特定特征表示"""
        logger.info(f"\n{'='*50}")
        logger.info(f"探索特征表示: {feature_type}")
        logger.info(f"{'='*50}")
        
        # 根据特征类型修改配置
        if feature_type == "raw":
            # 原始特征：x,y,z,l,w,h,ry
            pass
        elif feature_type == "normalized":
            # 归一化特征
            pass
        elif feature_type == "log_scale":
            # 对数尺度特征（用于尺寸）
            pass
        elif feature_type == "polar":
            # 极坐标表示（用于位置）
            pass
        
        # 训练模型
        train_loader, val_loader = build_dataloaders()
        model = build_model()
        trainer = Trainer(model, train_loader, val_loader)
        
        # 简化训练（少量epoch）
        original_epochs = cfg.EPOCHS
        cfg.EPOCHS = 10
        trainer.train()
        cfg.EPOCHS = original_epochs
        
        # 记录结果
        self.results[feature_type] = {
            'best_val_loss': trainer.best_val_loss
        }
        
        return self.results[feature_type]
    
    def explore_data_augmentation(self):
        """探索数据增强策略的影响"""
        logger.info(f"\n{'='*50}")
        logger.info("探索数据增强策略")
        logger.info(f"{'='*50}")
        
        augmentation_strategies = [
            "none",           # 无增强
            "flip_only",      # 仅翻转
            "color_only",     # 仅颜色
            "full"            # 完整增强
        ]
        
        aug_results = {}
        
        for strategy in augmentation_strategies:
            logger.info(f"\n测试增强策略: {strategy}")
            
            # 这里应该修改dataset.py中的增强策略
            # 简化演示，仅记录
            aug_results[strategy] = {
                'description': f'Augmentation: {strategy}',
                'val_loss': 0.0  # 实际应该训练并获取
            }
        
        return aug_results
    
    def explore_loss_weights(self):
        """探索损失函数权重的影响"""
        logger.info(f"\n{'='*50}")
        logger.info("探索损失函数权重")
        logger.info(f"{'='*50}")
        
        weight_configs = [
            {"location": 1.0, "dimension": 0.5, "orientation": 0.3},  # 默认
            {"location": 1.0, "dimension": 1.0, "orientation": 1.0},  # 均衡
            {"location": 2.0, "dimension": 0.5, "orientation": 0.3},  # 强调位置
            {"location": 1.0, "dimension": 1.0, "orientation": 0.5},  # 强调尺寸
            {"location": 1.0, "dimension": 0.5, "orientation": 1.0},  # 强调方向
        ]
        
        weight_results = {}
        
        for idx, weights in enumerate(weight_configs):
            logger.info(f"\n测试权重配置 {idx+1}: {weights}")
            
            # 修改配置
            original_weights = cfg.LOSS_WEIGHTS.copy()
            cfg.LOSS_WEIGHTS = weights
            
            # 训练（简化）
            train_loader, val_loader = build_dataloaders()
            model = build_model()
            trainer = Trainer(model, train_loader, val_loader)
            
            # 只训练几个epoch
            original_epochs = cfg.EPOCHS
            cfg.EPOCHS = 5
            trainer.train()
            cfg.EPOCHS = original_epochs
            
            weight_results[f"config_{idx+1}"] = {
                'weights': weights,
                'best_val_loss': trainer.best_val_loss
            }
            
            # 恢复原始权重
            cfg.LOSS_WEIGHTS = original_weights
        
        return weight_results
    
    def explore_network_depth(self):
        """探索网络深度的影响"""
        logger.info(f"\n{'='*50}")
        logger.info("探索网络深度")
        logger.info(f"{'='*50}")
        
        # 不同深度的模型
        models = ['simple', 'resnet', 'attention', 'transformer']
        
        depth_results = {}
        
        for model_type in models:
            logger.info(f"\n测试模型: {model_type}")
            
            original_type = cfg.BACKBONE_TYPE
            cfg.BACKBONE_TYPE = model_type
            
            model = build_model()
            param_count = sum(p.numel() for p in model.parameters())
            
            logger.info(f"参数数量: {param_count:,}")
            
            depth_results[model_type] = {
                'parameters': param_count,
                'description': f'{model_type} architecture'
            }
            
            cfg.BACKBONE_TYPE = original_type
        
        return depth_results
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        logger.info(f"\n{'='*50}")
        logger.info("分析特征重要性")
        logger.info(f"{'='*50}")
        
        # 加载数据
        train_loader, val_loader = build_dataloaders()
        
        # 收集所有目标值
        all_targets = []
        for _, targets, _ in val_loader:
            all_targets.append(targets.numpy())
        
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算每个特征的统计信息
        feature_names = ['x', 'y', 'z', 'length', 'width', 'height', 'rotation_y']
        
        importance_analysis = {}
        
        for idx, name in enumerate(feature_names):
            values = all_targets[:, idx]
            
            importance_analysis[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values)),
                'cv': float(np.std(values) / (np.mean(values) + 1e-6))  # 变异系数
            }
        
        # 打印分析结果
        logger.info("\n特征统计分析:")
        logger.info(f"{'Feature':<15} {'Mean':<10} {'Std':<10} {'Range':<10} {'CV':<10}")
        logger.info("-" * 60)
        for name, stats in importance_analysis.items():
            logger.info(f"{name:<15} {stats['mean']:<10.3f} {stats['std']:<10.3f} "
                       f"{stats['range']:<10.3f} {stats['cv']:<10.3f}")
        
        return importance_analysis
    
    def visualize_feature_exploration(self):
        """可视化特征探索结果"""
        vis_dir = os.path.join(cfg.VIS_DIR, 'feature_exploration')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. 特征重要性可视化
        importance = self.analyze_feature_importance()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        features = list(importance.keys())
        means = [importance[f]['mean'] for f in features]
        stds = [importance[f]['std'] for f in features]
        ranges = [importance[f]['range'] for f in features]
        cvs = [importance[f]['cv'] for f in features]
        
        axes[0, 0].bar(features, means, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Feature Means')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(features, stds, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('Feature Standard Deviations')
        axes[0, 1].set_ylabel('Std Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].bar(features, ranges, alpha=0.7, edgecolor='black', color='green')
        axes[1, 0].set_title('Feature Ranges')
        axes[1, 0].set_ylabel('Range')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(features, cvs, alpha=0.7, edgecolor='black', color='red')
        axes[1, 1].set_title('Coefficient of Variation')
        axes[1, 1].set_ylabel('CV')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"特征探索可视化保存到: {vis_dir}")
    
    def save_results(self):
        """保存探索结果"""
        results_file = os.path.join(cfg.LOG_DIR, 'feature_exploration_results.json')
        
        all_results = {
            'feature_representations': self.results,
            'feature_importance': self.analyze_feature_importance(),
            'network_depth': self.explore_network_depth()
        }
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info(f"特征探索结果保存到: {results_file}")

def main():
    """主函数"""
    logger.info("开始特征探索...")
    
    explorer = FeatureExplorer()
    
    # 1. 分析特征重要性
    logger.info("\n=== 特征重要性分析 ===")
    explorer.analyze_feature_importance()
    
    # 2. 探索网络深度
    logger.info("\n=== 网络深度探索 ===")
    explorer.explore_network_depth()
    
    # 3. 可视化
    logger.info("\n=== 生成可视化 ===")
    explorer.visualize_feature_exploration()
    
    # 4. 保存结果
    explorer.save_results()
    
    logger.info("\n特征探索完成！")

if __name__ == "__main__":
    main()
