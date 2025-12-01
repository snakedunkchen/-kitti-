import numpy as np
import torch
from config import cfg

def iou_3d(pred_box, target_box):
    """计算3D IoU（简化版）
    pred_box/target_box: [x,y,z,l,w,h,ry]
    """
    # 简化：计算BEV（鸟瞰图）IoU
    pred_x, pred_z = pred_box[0], pred_box[2]
    pred_l, pred_w = pred_box[3], pred_box[4]
    
    target_x, target_z = target_box[0], target_box[2]
    target_l, target_w = target_box[3], target_box[4]

    # 计算BEV边界框
    pred_min_x = pred_x - pred_l/2
    pred_max_x = pred_x + pred_l/2
    pred_min_z = pred_z - pred_w/2
    pred_max_z = pred_z + pred_w/2

    target_min_x = target_x - target_l/2
    target_max_x = target_x + target_l/2
    target_min_z = target_z - target_w/2
    target_max_z = target_z + target_w/2

    # 交并集计算
    inter_min_x = max(pred_min_x, target_min_x)
    inter_max_x = min(pred_max_x, target_max_x)
    inter_min_z = max(pred_min_z, target_min_z)
    inter_max_z = min(pred_max_z, target_max_z)

    inter_area = max(0, inter_max_x - inter_min_x) * max(0, inter_max_z - inter_min_z)
    pred_area = pred_l * pred_w
    target_area = target_l * target_w
    union_area = pred_area + target_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def compute_ap(predictions, targets, iou_threshold=0.5):
    """计算AP（平均精度）"""
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0

    # 计算IoU
    ious = []
    for pred in predictions:
        max_iou = 0.0
        for target in targets:
            iou = iou_3d(pred, target)
            if iou > max_iou:
                max_iou = iou
        ious.append(max_iou)

    # 计算TP/FP
    tp = np.array(ious) >= iou_threshold
    fp = ~tp

    # 按置信度排序（简化：假设置信度为1）
    tp = tp[np.argsort(-np.ones_like(ious))]
    fp = fp[np.argsort(-np.ones_like(ious))]

    # 计算累计TP/FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # 计算精度和召回率
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / len(targets)

    # 计算AP（11点插值法）
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        if np.sum(recall >= r) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= r])
        ap += p / 11.0

    return ap

class KITTIMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_predictions = []
        self.all_targets = []

    def update(self, preds, targets):
        """更新预测和目标"""
        self.all_predictions.extend(preds.detach().cpu().numpy())
        self.all_targets.extend(targets.detach().cpu().numpy())

    def compute(self):
        """计算所有评估指标"""
        metrics = {}
        for iou_thresh in cfg.IOU_THRESHOLDS:
            ap = compute_ap(self.all_predictions, self.all_targets, iou_thresh)
            metrics[f"AP_{iou_thresh}"] = ap
        
        # BEV AP
        for bev_iou_thresh in cfg.BEV_IOU_THRESHOLDS:
            bev_ap = compute_ap(self.all_predictions, self.all_targets, bev_iou_thresh)
            metrics[f"BEV_AP_{bev_iou_thresh}"] = bev_ap

        return metrics