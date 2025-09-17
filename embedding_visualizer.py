import numpy as np
import torch
from torch.nn.parallel import scatter

embedding_path = 'embedding/uda_daformer_HHHead_gta2cityscapes512_20250917_150026/frankfurt_000000_000294_leftImg8bit.pth'
gt_path = 'gt/uda_daformer_HHHead_gta2cityscapes512_20250917_150026/frankfurt_000000_000294_leftImg8bit.pth'

embedding = torch.load(embedding_path).cpu()
gt = torch.load(gt_path).cpu()
h, w = embedding.shape[-2:]
c = embedding.shape[1]
gt_resized = torch.nn.functional.interpolate(
    gt.unsqueeze(1).float(),
    size=(h,w),
    mode='nearest'
).squeeze(1).long()

features = embedding.squeeze(0)
gt_resized = gt_resized.squeeze(0)

features_flat = features.reshape(c, -1).T
gt_flat = gt_resized.reshape(-1)

unique_class = torch.unique(gt_flat)

samples_per_class = 500  # 每类采样500个点
sampled_features = []
sampled_labels = []

for cls in unique_class:
    if cls == 255:
        continue
    cls_mask = (gt_flat == cls)
    cls_features = features_flat[cls_mask]

    # 有更好的方法
    if len(cls_features) > samples_per_class:
        indices = torch.randperm(len(cls_features))[:samples_per_class]
        cls_features = cls_features[indices]

    sampled_features.append(cls_features)
    sampled_labels.extend([cls.item()] * len(cls_features))

sampled_features = torch.cat(sampled_features, dim=0).numpy()
sampled_labels = np.array(sampled_labels)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pca = PCA(n_components=50)
features_pca = pca.fit_transform(sampled_features)
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
features_tsne = tsne.fit_transform(features_pca)

plt.figure(figsize=(12,10))
scatter = plt.scatter(
    features_tsne[:, 0],
    features_tsne[:, 1],
    c=sampled_labels,
    cmap='tab20',
    alpha=0.6,
    s=10
)

plt.title('t-SNE Visualization of Feature Space by Class')
plt.colorbar(scatter, label='Class Label')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)

# 保存图像
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()




