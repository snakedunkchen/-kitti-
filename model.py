import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

# ===================== 基础模块 =====================
class ResBlock(nn.Module):
    """残差块：解决深层梯度消失"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ChannelAttention(nn.Module):
    """通道注意力：强化关键特征通道"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    """空间注意力：强化关键空间位置"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ===================== 模型定义 =====================
class SimpleMono3DCNN(nn.Module):
    """基础轻量CNN"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(cfg.INPUT_CHANNELS, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.PRED_DIM)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regression_head(features)

class ResNetMono3D(nn.Module):
    """残差版CNN"""
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(cfg.INPUT_CHANNELS, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(ResBlock, 64, 2, 2)
        self.layer2 = self._make_layer(ResBlock, 128, 2, 2)
        self.layer3 = self._make_layer(ResBlock, 256, 2, 2)
        self.layer4 = self._make_layer(ResBlock, 512, 2, 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.PRED_DIM)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        return self.regression_head(out)

class AttentionMono3D(nn.Module):
    """注意力增强版CNN"""
    def __init__(self):
        super().__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 注意力模块
        self.attention = ChannelAttention(512)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 回归头
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.PRED_DIM)
        )

    def forward(self, x):
        features = self.features(x)
        features = self.attention(features)
        features = self.avg_pool(features)
        return self.regression_head(features)

class TransformerMono3D(nn.Module):
    """Transformer增强版CNN：结合CNN特征提取和Transformer全局建模"""
    def __init__(self):
        super().__init__()
        # CNN特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512),
        )

        # Transformer编码器
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, 1, 1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 回归头
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.PRED_DIM)
        )

    def forward(self, x):
        # CNN特征提取
        features = self.features(x)
        b, c, h, w = features.shape
        
        # 添加位置编码
        features = features + self.pos_embedding
        
        # 重塑为序列格式 (batch, seq_len, channels)
        features_seq = features.flatten(2).permute(0, 2, 1)
        
        # Transformer编码
        features_seq = self.transformer(features_seq)
        
        # 全局平均池化
        features = features_seq.mean(dim=1)
        
        # 回归预测
        return self.regression_head(features)

class EnsembleMono3D(nn.Module):
    """集成模型：结合多个模型的预测"""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 加权平均
        weights = F.softmax(self.weights, dim=0)
        ensemble_output = sum(w * out for w, out in zip(weights, outputs))
        return ensemble_output

# 构建模型
def build_model():
    if cfg.BACKBONE_TYPE == "simple":
        model = SimpleMono3DCNN()
    elif cfg.BACKBONE_TYPE == "resnet":
        model = ResNetMono3D()
    elif cfg.BACKBONE_TYPE == "attention":
        model = AttentionMono3D()
    elif cfg.BACKBONE_TYPE == "transformer":
        model = TransformerMono3D()
    elif cfg.BACKBONE_TYPE == "ensemble":
        # 创建集成模型
        models = [
            SimpleMono3DCNN(),
            ResNetMono3D(),
            AttentionMono3D()
        ]
        model = EnsembleMono3D(models)
    else:
        raise ValueError(f"不支持的模型类型: {cfg.BACKBONE_TYPE}")
    
    # 加载预训练权重
    if cfg.LOAD_CHECKPOINT and os.path.exists(cfg.CHECKPOINT_PATH):
        checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"加载预训练权重: {cfg.CHECKPOINT_PATH}")
    
    return model.to(cfg.DEVICE)