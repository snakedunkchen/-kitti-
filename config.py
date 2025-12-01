import os
import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    # ===================== 路径配置 =====================
    ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT: str = os.path.join(ROOT_DIR, "data/kitti")
    CHECKPOINT_DIR: str = os.path.join(ROOT_DIR, "checkpoints")
    LOG_DIR: str = os.path.join(ROOT_DIR, "logs")
    VIS_DIR: str = os.path.join(ROOT_DIR, "visualizations")

    # ===================== 数据配置 =====================
    IMAGE_SIZE: tuple = (375, 1242)  # KITTI原始尺寸 (H, W)
    INPUT_CHANNELS: int = 3
    NUM_CLASSES: int = 1  # 仅检测Car类
    PRED_DIM: int = 7     # 3D参数：x,y,z,l,w,h,ry（ry=偏航角）
    TRAIN_VAL_SPLIT: float = 0.8  # 训练/验证拆分比例
    MEAN: list = field(default_factory=lambda: [123.68, 116.779, 103.939])  # ImageNet均值
    STD: list = field(default_factory=lambda: [58.393, 57.12, 57.375])      # ImageNet标准差

    # ===================== 训练配置 =====================
    BATCH_SIZE: int = 4  # KITTI图像大，建议小批量
    EPOCHS: int = 50
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED: int = 42

    # 训练增强配置
    GRAD_CLIP_NORM: float = 5.0  # 梯度裁剪阈值
    EARLY_STOP_PATIENCE: int = 8  # 早停轮数
    LR_SCHEDULER_PATIENCE: int = 3  # 学习率衰减耐心值
    LR_SCHEDULER_FACTOR: float = 0.5  # 学习率衰减因子

    # ===================== 模型配置 =====================
    BACKBONE_TYPE: str = "attention"  # 可选：simple / resnet / attention
    LOAD_CHECKPOINT: bool = False
    CHECKPOINT_PATH: str = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # ===================== 损失配置 =====================
    LOSS_WEIGHTS: dict = field(default_factory=lambda: {
        "location": 1.0,
        "dimension": 0.5,
        "orientation": 0.3
    })

    # ===================== 评估配置 =====================
    IOU_THRESHOLDS: list = field(default_factory=lambda: [0.5, 0.7])
    BEV_IOU_THRESHOLDS: list = field(default_factory=lambda: [0.5, 0.7])
    VIS_EVERY_EPOCH: int = 5  # 每N轮可视化一次

    def __post_init__(self):
        # 创建必要目录
        for dir in [self.CHECKPOINT_DIR, self.LOG_DIR, self.VIS_DIR]:
            os.makedirs(dir, exist_ok=True)

# 全局配置实例
cfg = Config()

# 固定随机种子
def set_seed(seed: int = cfg.SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed()