# [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

* Year: 25 Mar `2021`
* Author: Ze Liu
* Abstract: This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \textbf{S}hifted \textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at [this https URL](https://github.com/microsoft/Swin-Transformer).

----------------------------------------------------------------------------------------------------

## 1. Introduction

### 3.2 Shifted Window based Self-Attention

> **Shifted window partitioning in successive blocks** The window-based self-attention module lacks connections across windows, which limits its modeling power. To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, we propose a shifted window partitioning approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks.

> $$\hat{\textbf{z}}^{l} = \text{W-MSA}(\text{LN}(\textbf{z}^{l-1})) + \textbf{z}^{l-1},$$
> $$\textbf{z}^{l} = \text{MLP}(\text{LN}(\hat{\textbf{z}}^{l})) + \hat{\textbf{z}}^{l},$$
> $$\hat{\textbf{z}}^{l+1} = \text{SW-MSA}(\text{LN}(\textbf{z}^{l})) + \textbf{z}^{l},$$
> $$\textbf{z}^{l+1} = \text{MLP}(\text{LN}(\hat{\textbf{z}}^{l+1})) + \hat{\textbf{z}}^{l+1}, \tag{3}$$
where $\hat{\textbf{z}}^{l}$ and $\textbf{z}^{l}$ denote the output features of the (S)W-MSA module and the MLP module for block $l$, respectively; W-MSA and SW-MSA denote window based multi-head self-attention using regular and shifted window partitioning configurations, respectively.

> The shifted window partitioning approach introduces connections between neighboring non-overlapping windows in the previous layer and is found to be effective in image classification, object detection, and semantic segmentation.
