from config import cfg
from dataset import build_dataloaders
from model import build_model
from trainer import Trainer
from utils import logger

def main():
    # 1. 构建数据加载器
    logger.info("构建数据加载器...")
    train_loader, val_loader = build_dataloaders()
    
    # 2. 构建模型
    logger.info(f"构建模型: {cfg.BACKBONE_TYPE}")
    model = build_model()
    logger.info(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 构建训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()

if __name__ == "__main__":
    main()