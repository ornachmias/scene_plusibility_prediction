import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

vit = [[343, 95, 43, 38, 80, 87, 48]
    , [198, 266, 58, 38, 33, 95, 46]
    , [144, 113, 261, 22, 39, 121, 34]
    , [103, 91, 15, 400, 24, 78, 23]
    , [272, 104, 59, 28, 133, 88, 50]
    , [228, 74, 33, 37, 45, 273, 44]
    , [292, 62, 34, 55, 44, 85, 162]]
vit = np.array(vit)
vit = vit / vit.sum(axis=1, keepdims=True)

resnet = [[427, 81, 24, 38, 68, 35, 61]
    , [147, 397, 18, 47, 34, 49, 42]
    , [103, 60, 465, 17, 57, 12, 20]
    , [100, 40, 5, 555, 10, 7, 17]
    , [172, 124, 46, 18, 310, 32, 32]
    , [165, 35, 3, 26, 18, 449, 38]
    , [236, 77, 43, 14, 54, 39, 271]]
resnet = np.array(resnet)
resnet = resnet / resnet.sum(axis=1, keepdims=True)

crtnet = [[481, 51, 36, 56, 21, 27, 62]
    , [108, 442, 55, 60, 11, 21, 37]
    , [43, 66, 566, 7, 19, 13, 20]
    , [44, 59, 6, 569, 3, 20, 33]
    , [157, 117, 63, 10, 334, 22, 31]
    , [121, 58, 6, 14, 17, 464, 54]
    , [121, 55, 34, 49, 24, 39, 412]]
crtnet = np.array(crtnet)
crtnet = crtnet / crtnet.sum(axis=1, keepdims=True)

coatnet = [[277, 113, 58, 44, 53, 73, 116]
    , [124, 253, 55, 49, 50, 68, 135]
    , [64, 77, 352, 10, 41, 89, 101]
    , [80, 62, 28, 453, 12, 49, 50]
    , [166, 153, 64, 7, 163, 90, 91]
    , [166, 98, 75, 16, 35, 299, 45]
    , [180, 62, 42, 61, 49, 73, 267]]
coatnet = np.array(coatnet)
coatnet = coatnet / coatnet.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

labels = ['Plausible', 'Location', 'Rotation', 'Gravity', 'Intersection', 'Size', 'Pose']

sns.heatmap(vit, annot=True, xticklabels=labels, yticklabels=labels, fmt='.2f', ax=axes[0, 0])
axes[0, 0].set_title('ViT')
sns.heatmap(resnet, annot=True, xticklabels=labels, yticklabels=labels, fmt='.2f', ax=axes[0, 1])
axes[0, 1].set_title('ResNet')
sns.heatmap(crtnet, annot=True, xticklabels=labels, yticklabels=labels, fmt='.2f', ax=axes[1, 0])
axes[1, 0].set_title('CRTNet')
sns.heatmap(coatnet, annot=True, xticklabels=labels, yticklabels=labels, fmt='.2f', ax=axes[1, 1])
axes[1, 1].set_title('CoAtNet')

plt.tight_layout()
plt.savefig('../all_cf_v2.png')



