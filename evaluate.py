"""
综合评估脚本：评估多个模型并生成详细报告
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from config import cfg
from dataset import build_dataloaders
from model import build_model
from metrics import KITTIMetrics, iou_3d
from utils import logger
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model_type, checkpoint_path=None):
        """评估单个模型"""
        logger.info(f"\n{'='*50}")
        logger.info(f"评估模型: {model_type}")
        logger.info(f"{'='*50}")
        
        # 设置模型类型
        original_type = cfg.BACKBONE_TYPE
        cfg.BACKBONE_TYPE = model_type
        
        # 构建模型
        model = build_model()
        
        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"加载检查点: {checkpoint_path}")
        else:
            logger.warning(f"检查点不存在: {checkpoint_path}")
        
        model.eval()
        
        # 构建数据加载器
        _, val_loader = build_dataloaders()
        
        # 评估指标
        metrics = KITTIMetrics()
        
        # 收集预测和目标
        all_preds = []
        all_targets = []
        all_errors = {
            'location': [],
            'dimension': [],
            'orientation': []
        }
        
        inference_times = []
        
        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc=f"Evaluating {model_type}"):
                images = images.to(cfg.DEVICE)
                targets = targets.to(cfg.DEVICE)
                
                # 推理计时
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                preds = model(images)
                end_time.record()
                
                torch.cuda.synchronize()
                inference_times.append(start_time.elapsed_time(end_time))
                
                # 更新指标
                metrics.update(preds, targets)
                
                # 收集数据
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                all_preds.extend(preds_np)
                all_targets.extend(targets_np)
                
                # 计算误差
                loc_error = np.abs(preds_np[:, :3] - targets_np[:, :3])
                dim_error = np.abs(preds_np[:, 3:6] - targets_np[:, 3:6])
                orient_error = np.abs(preds_np[:, 6] - targets_np[:, 6])
                
                all_errors['location'].extend(loc_error)
                all_errors['dimension'].extend(dim_error)
                all_errors['orientation'].extend(orient_error)
        
        # 计算评估指标
        eval_metrics = metrics.compute()
        
        # 计算额外统计信息
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 平均误差
        mae_location = np.mean(np.abs(all_preds[:, :3] - all_targets[:, :3]), axis=0)
        mae_dimension = np.mean(np.abs(all_preds[:, 3:6] - all_targets[:, 3:6]), axis=0)
        mae_orientation = np.mean(np.abs(all_preds[:, 6] - all_targets[:, 6]))
        
        # RMSE
        rmse_location = np.sqrt(np.mean((all_preds[:, :3] - all_targets[:, :3])**2, axis=0))
        rmse_dimension = np.sqrt(np.mean((all_preds[:, 3:6] - all_targets[:, 3:6])**2, axis=0))
        rmse_orientation = np.sqrt(np.mean((all_preds[:, 6] - all_targets[:, 6])**2))
        
        # 推理速度
        avg_inference_time = np.mean(inference_times)
        fps = 1000.0 / avg_inference_time  # 转换为FPS
        
        # 保存结果
        results = {
            'model_type': model_type,
            'metrics': eval_metrics,
            'mae': {
                'location': mae_location.tolist(),
                'dimension': mae_dimension.tolist(),
                'orientation': float(mae_orientation)
            },
            'rmse': {
                'location': rmse_location.tolist(),
                'dimension': rmse_dimension.tolist(),
                'orientation': float(rmse_orientation)
            },
            'inference': {
                'avg_time_ms': float(avg_inference_time),
                'fps': float(fps)
            },
            'predictions': all_preds.tolist()[:100],  # 保存前100个预测
            'targets': all_targets.tolist()[:100]
        }
        
        self.results[model_type] = results
        
        # 打印结果
        logger.info(f"\n=== {model_type} 评估结果 ===")
        for k, v in eval_metrics.items():
            logger.info(f"{k}: {v:.4f}")
        logger.info(f"MAE Location (x,y,z): {mae_location}")
        logger.info(f"MAE Dimension (l,w,h): {mae_dimension}")
        logger.info(f"MAE Orientation: {mae_orientation:.4f}")
        logger.info(f"Inference Time: {avg_inference_time:.2f}ms ({fps:.1f} FPS)")
        
        # 恢复原始配置
        cfg.BACKBONE_TYPE = original_type
        
        return results
    
    def compare_models(self):
        """比较多个模型"""
        if len(self.results) < 2:
            logger.warning("需要至少2个模型进行比较")
            return
        
        logger.info(f"\n{'='*50}")
        logger.info("模型对比分析")
        logger.info(f"{'='*50}")
        
        # 创建对比表格
        comparison_data = []
        for model_type, results in self.results.items():
            row = {
                'Model': model_type,
                'AP_0.5': results['metrics'].get('AP_0.5', 0),
                'AP_0.7': results['metrics'].get('AP_0.7', 0),
                'MAE_Loc': np.mean(results['mae']['location']),
                'MAE_Dim': np.mean(results['mae']['dimension']),
                'MAE_Orient': results['mae']['orientation'],
                'FPS': results['inference']['fps']
            }
            comparison_data.append(row)
        
        # 打印对比表格
        logger.info("\n模型性能对比:")
        logger.info(f"{'Model':<15} {'AP@0.5':<10} {'AP@0.7':<10} {'MAE_Loc':<10} {'MAE_Dim':<10} {'MAE_Orient':<12} {'FPS':<10}")
        logger.info("-" * 90)
        for row in comparison_data:
            logger.info(f"{row['Model']:<15} {row['AP_0.5']:<10.4f} {row['AP_0.7']:<10.4f} "
                       f"{row['MAE_Loc']:<10.4f} {row['MAE_Dim']:<10.4f} "
                       f"{row['MAE_Orient']:<12.4f} {row['FPS']:<10.1f}")
        
        return comparison_data
    
    def visualize_results(self):
        """可视化评估结果"""
        if not self.results:
            logger.warning("没有评估结果可视化")
            return
        
        # 创建可视化目录
        vis_dir = os.path.join(cfg.VIS_DIR, 'evaluation')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. AP对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(self.results.keys())
        ap_05 = [self.results[m]['metrics'].get('AP_0.5', 0) for m in models]
        ap_07 = [self.results[m]['metrics'].get('AP_0.7', 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, ap_05, width, label='AP@0.5', alpha=0.8)
        axes[0].bar(x + width/2, ap_07, width, label='AP@0.7', alpha=0.8)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Average Precision')
        axes[0].set_title('Model Performance Comparison - AP')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 推理速度对比
        fps = [self.results[m]['inference']['fps'] for m in models]
        axes[1].bar(models, fps, alpha=0.8, color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('FPS')
        axes[1].set_title('Inference Speed Comparison')
        axes[1].set_xticklabels(models, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. MAE对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        mae_loc = [np.mean(self.results[m]['mae']['location']) for m in models]
        mae_dim = [np.mean(self.results[m]['mae']['dimension']) for m in models]
        mae_orient = [self.results[m]['mae']['orientation'] for m in models]
        
        axes[0].bar(models, mae_loc, alpha=0.8, color='lightcoral', edgecolor='black')
        axes[0].set_ylabel('MAE (meters)')
        axes[0].set_title('Location Error')
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].bar(models, mae_dim, alpha=0.8, color='lightgreen', edgecolor='black')
        axes[1].set_ylabel('MAE (meters)')
        axes[1].set_title('Dimension Error')
        axes[1].set_xticklabels(models, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].bar(models, mae_orient, alpha=0.8, color='lightyellow', edgecolor='black')
        axes[2].set_ylabel('MAE (radians)')
        axes[2].set_title('Orientation Error')
        axes[2].set_xticklabels(models, rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'error_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化结果保存到: {vis_dir}")
    
    def save_results(self):
        """保存评估结果到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(cfg.LOG_DIR, f'evaluation_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        logger.info(f"评估结果保存到: {results_file}")
        return results_file

def main():
    """主评估流程"""
    logger.info("开始模型评估...")
    
    evaluator = ModelEvaluator()
    
    # 评估多个模型
    models_to_evaluate = [
        ('simple', os.path.join(cfg.CHECKPOINT_DIR, 'simple_best.pth')),
        ('resnet', os.path.join(cfg.CHECKPOINT_DIR, 'resnet_best.pth')),
        ('attention', os.path.join(cfg.CHECKPOINT_DIR, 'attention_best.pth')),
        ('transformer', os.path.join(cfg.CHECKPOINT_DIR, 'transformer_best.pth')),
    ]
    
    for model_type, checkpoint_path in models_to_evaluate:
        try:
            evaluator.evaluate_model(model_type, checkpoint_path)
        except Exception as e:
            logger.error(f"评估 {model_type} 时出错: {str(e)}")
            continue
    
    # 模型对比
    if len(evaluator.results) > 1:
        evaluator.compare_models()
        evaluator.visualize_results()
    
    # 保存结果
    evaluator.save_results()
    
    logger.info("\n评估完成！")

if __name__ == "__main__":
    main()
