import os
import os.path as opt

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def align_gt_to_feature(gt, target_shape):
    """将标注图下采样至特征图的空间尺寸（必须用最近邻插值）"""
    return F.interpolate(
        gt.unsqueeze(1).float(),
        size=target_shape,
        mode='nearest'
    ).squeeze(1).long()


def compute_class_prototypes(features, gt_resized, class_labels):
    """
    计算每个类别的原型（均值向量）
    Args:
        features: (C, H, W) 特征图
        gt_resized: (H, W) 对齐后的标注图
        class_labels: list[int] 所有类别标签（如 [0,1,2,...]）
    Returns:
        prototypes: dict{ class_id: (C,) } 每个类别的原型向量
        counts: dict{ class_id: int } 每个类别的像素数量
    """
    prototypes, counts = {}, {}
    for c in class_labels:
        mask = (gt_resized == c)
        if mask.sum() == 0:  # 忽略没有出现的类别
            continue
        prototypes[c] = features[:,mask].mean(dim=1)  # (C,)
        counts[c] = mask.sum().item()
    return prototypes, counts


def compute_similarity_matrix(prototypes, metric='cosine'):
    """
    计算类别间的相似度矩阵
    Args:
        prototypes: dict{ class_id: (C,) }
        metric: 'cosine' 或 'euclidean'
    Returns:
        sim_matrix: (n_classes, n_classes) 相似度矩阵
        class_ids: list[int] 对应的类别标签
    """
    class_ids = sorted(prototypes.keys())
    feat_vectors = torch.stack([prototypes[c] for c in class_ids])  # (n_classes, C)

    if metric == 'cosine':
        sim_matrix = cosine_similarity(feat_vectors.cpu().numpy())
    elif metric == 'euclidean':
        dist_matrix = torch.cdist(feat_vectors, feat_vectors).cpu().numpy()
        sim_matrix = 1 / (1 + dist_matrix)  # 将距离转换为相似度
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return sim_matrix, class_ids


def plot_similarity_matrix(sim_matrix, class_ids, class_names=None, sample_name=None):
    """用热力图可视化相似度矩阵"""
    if class_names is None:
        class_names = [str(c) for c in class_ids]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True, fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="YlOrRd", vmin=0, vmax=1
    )
    plt.title("Class Similarity Matrix (Cosine)")
    if sample_name:
        os.makedirs('cosine_sim', exist_ok=True)
        name = opt.join('cosine_sim', sample_name.split('.')[0])+'.png'
    else:
        name = 'cosine_similarity.png'
    plt.savefig(name, dpi=300)



def quantitative_analysis(features, gt, class_labels, class_names=None, sample_name=None):
    """完整的定量分析流程"""
    # Step 1: 数据对齐
    gt_resized = align_gt_to_feature(gt, (features.shape[1], features.shape[2]))
    gt_resized = gt_resized.squeeze()

    # Step 2: 计算类别原型
    prototypes, counts = compute_class_prototypes(features, gt_resized, class_labels)
    print(f"Class counts: {counts}")

    # Step 3: 计算相似度矩阵
    sim_matrix, class_ids = compute_similarity_matrix(prototypes, metric='cosine')

    # Step 4: 可视化
    plot_similarity_matrix(sim_matrix, class_ids, class_names, sample_name)

    return sim_matrix, prototypes, counts


if __name__ == '__main__':
    labels_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic light', 'traffic sign', 'vegetation', 'terrain',
                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle']
    label_dic = {}
    for i in range(len(labels_names)):
        label_dic[i] = labels_names[i]

    # embedding_path = 'embedding/uda_daformer_HHHead_gta2cityscapes512_20250917_150026/frankfurt_000000_000294_leftImg8bit.pth'
    # gt_path = 'gt/uda_daformer_HHHead_gta2cityscapes512_20250917_150026/frankfurt_000000_000294_leftImg8bit.pth'
    embedding_root = 'embedding/uda_daformer_HHHead_gta2cityscapes512_20250917_150026'
    gt_root = 'gt/uda_daformer_HHHead_gta2cityscapes512_20250917_150026'

    sample_dir = os.listdir(embedding_root)
    for name in sample_dir:
        embedding_path = opt.join(embedding_root, name)
        gt_path = opt.join(gt_root, name)

        embedding = torch.load(embedding_path).cpu()
        gt = torch.load(gt_path).cpu()

        gt_unique = torch.unique(gt)
        gt_unique = [i for i in gt_unique if i != 255]
        gt_name = [labels_names[i] for i in gt_unique if i != 255]

        quantitative_analysis(
            embedding[0],
            gt,
            gt_unique,
            gt_name,
            sample_name=name
        )
