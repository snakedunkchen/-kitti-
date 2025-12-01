import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from config import cfg

# ===================== 数据增强/变换 =====================
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, target=None):
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size  # (H, W)

    def __call__(self, image, target=None):
        image = cv2.resize(image, (self.size[1], self.size[0]))
        return image, target

class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = cv2.flip(image, 1)
            # 翻转时偏航角取反
            if target is not None and len(target) > 0:
                target[6] = -target[6]
        return image, target

class RandomBrightnessContrast:
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image, target=None):
        # 亮度调整
        brightness = np.random.uniform(*self.brightness_range)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # 对比度调整
        contrast = np.random.uniform(*self.contrast_range)
        image = np.clip((image - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
        
        return image, target

def build_transforms(train: bool = True):
    transforms = []
    if train:
        transforms.extend([
            RandomFlip(prob=0.5),
            RandomBrightnessContrast()
        ])
    transforms.extend([
        Resize(cfg.IMAGE_SIZE),
        Normalize(mean=cfg.MEAN, std=cfg.STD)
    ])
    return Compose(transforms)

# ===================== KITTI数据集 =====================
class KITTIDataset(Dataset):
    def __init__(self, root: str, split: str = "training", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # 路径配置
        self.image_dir = os.path.join(root, split, "image_2")
        self.label_dir = os.path.join(root, split, "label_2")

        # 兼容某些数据解压后出现的冗余目录结构，例如 data/kitti/training/training/*
        if not os.path.isdir(self.image_dir) and os.path.isdir(os.path.join(root, split, split, "image_2")):
            self.image_dir = os.path.join(root, split, split, "image_2")
        if not os.path.isdir(self.label_dir) and os.path.isdir(os.path.join(root, split, split, "label_2")):
            self.label_dir = os.path.join(root, split, split, "label_2")
        
        # 过滤有效图像文件
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith((".png", ".jpg"))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def _parse_label(self, label_path: str):
        """解析KITTI标注，提取Car类的3D参数"""
        targets = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip().split()
                if len(line) < 15 or line[0] != "Car":
                    continue
                # 提取3D参数：x,y,z,l,w,h,ry
                x, y, z = float(line[11]), float(line[12]), float(line[13])
                l, w, h = float(line[10]), float(line[9]), float(line[8])  # KITTI: h,w,l → l,w,h
                ry = float(line[14])  # 偏航角
                targets.append([x, y, z, l, w, h, ry])
        
        # 无Car时返回全0
        if not targets:
            return np.zeros(7, dtype=np.float32)
        # 简化：取第一个Car目标
        return np.array(targets[0], dtype=np.float32)

    def __getitem__(self, idx):
        # 1. 加载图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"图像加载失败: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 加载标注
        label_name = img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        label_path = os.path.join(self.label_dir, label_name)
        target = self._parse_label(label_path)

        # 3. 数据增强/变换
        if self.transform:
            image, target = self.transform(image, target)

        # 4. 格式转换
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        target = torch.from_numpy(target).float()

        return image, target, img_name

# 构建数据加载器
def build_dataloaders():
    # 完整数据集
    full_dataset = KITTIDataset(
        root=cfg.DATA_ROOT,
        split="training",
        transform=build_transforms(train=True)
    )
    
    # 拆分训练/验证集
    train_size = int(cfg.TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 验证集关闭数据增强
    val_dataset.dataset.transform = build_transforms(train=False)

    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader