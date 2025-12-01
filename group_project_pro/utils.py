import os
import cv2
import numpy as np
import torch
import logging
from config import cfg

# ===================== 日志工具 =====================
def setup_logger():
    """设置日志记录"""
    logger = logging.getLogger("mono3d_det")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    log_file = os.path.join(cfg.LOG_DIR, "train.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ===================== 3D框可视化 =====================
def project_3d_box_to_2d(box_3d, calib_matrix):
    """将3D框投影到2D图像
    box_3d: [x,y,z,l,w,h,ry]
    calib_matrix: 相机内参矩阵 (3,3)
    """
    x, y, z, l, w, h, ry = box_3d
    
    # 3D框8个顶点
    vertices = np.array([
        [l/2, w/2, h/2], [l/2, -w/2, h/2], [-l/2, -w/2, h/2], [-l/2, w/2, h/2],
        [l/2, w/2, -h/2], [l/2, -w/2, -h/2], [-l/2, -w/2, -h/2], [-l/2, w/2, -h/2]
    ])
    
    # 旋转（绕y轴）
    rot_mat = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    vertices = vertices @ rot_mat.T
    
    # 平移到相机坐标系
    vertices += np.array([x, y, z])
    
    # 投影到2D
    vertices_2d = vertices @ calib_matrix.T
    vertices_2d = vertices_2d[:, :2] / vertices_2d[:, 2:3]
    
    return vertices_2d.astype(np.int32)

def visualize_3d_box(image, box_3d, calib_matrix, color=(0, 255, 0), thickness=2):
    """可视化3D框"""
    vertices_2d = project_3d_box_to_2d(box_3d, calib_matrix)
    
    # 绘制3D框线
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    for (i, j) in edges:
        cv2.line(image, tuple(vertices_2d[i]), tuple(vertices_2d[j]), color, thickness)
    
    return image

# ===================== 模型加载/保存 =====================
def save_checkpoint(model, optimizer, epoch, val_loss, metrics):
    """保存模型检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "metrics": metrics
    }
    torch.save(checkpoint, cfg.CHECKPOINT_PATH)
    logger.info(f"保存模型检查点: {cfg.CHECKPOINT_PATH}")

def load_checkpoint(model, optimizer=None):
    """加载模型检查点"""
    if not os.path.exists(cfg.CHECKPOINT_PATH):
        logger.warning("检查点文件不存在，加载失败")
        return 0, float("inf")
    
    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    logger.info(f"加载检查点：epoch={epoch}, val_loss={val_loss:.4f}")
    return epoch, val_loss