# [Notes][Vision][Transformer] PVT

* url: https://arxiv.org/abs/2102.12122
* Year: 24 Feb `2021`
* Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao
* Abstract: Although using convolutional neural networks (CNNs) as backbones achieves great successes in computer vision, this work investigates a simple backbone network useful for many dense prediction tasks without convolutions. Unlike the recently-proposed Transformer model (e.g., ViT) that is specially designed for image classification, we propose Pyramid Vision Transformer~(PVT), which overcomes the difficulties of porting Transformer to various dense prediction tasks. PVT has several merits compared to prior arts. (1) Different from ViT that typically has low-resolution outputs and high computational and memory cost, PVT can be not only trained on dense partitions of the image to achieve high output resolution, which is important for dense predictions but also using a progressive shrinking pyramid to reduce computations of large feature maps. (2) PVT inherits the advantages from both CNN and Transformer, making it a unified backbone in various vision tasks without convolutions by simply replacing CNN backbones. (3) We validate PVT by conducting extensive experiments, showing that it boosts the performance of many downstream tasks, e.g., object detection, semantic, and instance segmentation. For example, with a comparable number of parameters, RetinaNet+PVT achieves 40.4 AP on the COCO dataset, surpassing RetinaNet+ResNet50 (36.3 AP) by 4.1 absolute AP. We hope PVT could serve as an alternative and useful backbone for pixel-level predictions and facilitate future researches. Code is available at [this https URL](https://github.com/whai362/PVT).

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* > This work proposes a convolution-free backbone network using Transformer model.
* Proposed progressive shrinking pyramid.
* Proposed spatial reduction attention (SRA) to replace the traditional multi-head attention (MHA).

----------------------------------------------------------------------------------------------------

## 1. Introduction

> Our PVT overcomes the difficulties of the conventional Transformer by (1) taking fine-grained image patches (i.e., $4 \times 4$ pixels per patch) as input to learn high-resolution representation, which is essential for dense prediction tasks; (2) introducing a progressive shrinking pyramid to reduce the sequence length of Transformer as the network deepens, significantly reducing the computational cost, and (3) adopting a spatial-reduction attention (SRA) layer to further reduce the resource consumption when learning high-resolution features.

> Overall, the proposed PVT possesses the following merits.
> 1. compared to the traditional CNN backbones, which have local receptive fields that increase with the network depth, our PVT always produces a global receptive field, which is more suitable for detection and segmentation.
> 2. compared to ViT, thanks to its advanced pyramid structure, our method can more easily be plugged into many representative dense prediction pipelines, e.g., RetinaNet and Mask R-CNN.
> 3. we can build a convolution-free pipeline by combining our PVT with other task-specific Transformer decoders, such as PVT+DETR for object detection.

## 2. Related Work

## 3. Pyramid Vision Transformer (PVT)

### 3.1 Overall Architecture

> Our goal is to introduce the pyramid structure into Transformer, so that it can generate multi-scale feature maps for dense prediction tasks (e.g., object detection and semantic segmentation).

### 3.2. Feature Pyramid for Transformer

> Unlike CNN backbone networks [15] that use convolution stride to obtain multi-scale feature maps, our PVT use `progressive shrinking strategy` to control the scale of feature maps by patch embedding layers.

Notations:
* Let $P_{i} \in \mathbb{Z}_{++}$ denote the side length of the patches in stage $i$.
* Let $C_{i} \in \mathbb{Z}_{++}$ denote the dimension of the patch embedding in stage $i$.
* Let $H_{i}, W_{i} \in \mathbb{Z}_{++}$ denote the height and width of the output feature map of stage $i$.
* Let $F_{i} \in \mathbb{R}^{H_{i} \times W_{i} \times C_{i}}$ denote the output feature map of stage $i$.

Then patch embedding is the following process:

$$\mathbb{R}^{H_{i-1} \times W_{i-1} \times C_{i-1}} \overset{\text{divide}}{\to}
\mathbb{R}^{\frac{H_{i-1}}{P_{i}} \times \frac{W_{i-1}}{P_{i}} \times P_{i} \times P_{i} \times C_{i-1}} \overset{\text{flatten}}{\to}
\mathbb{R}^{\frac{H_{i-1}}{P_{i}} \times \frac{W_{i-1}}{P_{i}} \times P_{i}^{2}C_{i-1}} \overset{\text{project}}{\to}
\mathbb{R}^{\frac{H_{i-1}}{P_{i}} \times \frac{W_{i-1}}{P_{i}} \times C_{i}}$$

### 3.3. Transformer Encoder

> We propose a spatial-reduction attention (SRA) layer to replace the traditional multi-head attention (MHA) layer [51] in the encoder.

Notations:
* Let $L_{i} \in \mathbb{Z}_{++}$ denote the number of encoder layers in stage $i$.
* Let $Q \in \mathbb{R}^{}$ denote the input query to the SRA layer.
* Let $K \in \mathbb{R}^{}$ denote the input key to the SRA layer.
* Let $V \in \mathbb{R}^{}$ denote the input value to the SRA layer.
* Let $N_{i} \in \mathbb{Z}_{++}$ denote the number of heads of the SRA layers in stage $i$.
* Let $d \in \mathbb{Z}_{++}$ denote the dimension of the heads of the SRA layers in stage $i$.
Then we have $d = C_{i} / N_{i}$.
* Let $W_{j}^{Q}, W_{j}^{K}, W_{j}^{V} \in \mathbb{R}^{C_{i} \times d}$ denote the projection matrices in head $j$, stage $i$.
* Let $W^{O} \in \mathbb{R}^{C_{i} \times C_{i}}$ denote the output projection matrix in stage $i$.
* Let $W^{S} \in \mathbb{R}^{R_{i}^{2}C_{i} \times C_{i}}$ the spacial reduction projection matrix in stage $i$.

Then the SRA layer is defined by the following equations:

$$\operatorname{SRA}(Q, K, V) := \operatorname{Concatenate}\bigg\{\text{head}_{j}: j \in \{1, ..., N_{i}\bigg\}W^{O},$$
$$\text{head}_{j} := \operatorname{Attention}(QW_{j}^{Q}, \operatorname{SR}(K)W_{j}^{K}, \operatorname{SR}(V)W_{j}^{V}),$$
$$\operatorname{SR}(x) := \operatorname{LayerNormalization}(\operatorname{Reshape}(x, R_{i})W^{S}),$$
$$\operatorname{Reshape}(R_{i}): \mathbb{R}^{H_{i} \times W_{i} \times C_{i}} \to \mathbb{R}^{\frac{H_{i}W_{i}}{R_{i}^{2}} \times R_{i}^{2}C_{i}}.$$

### 3.5. Discussion

> Both PVT and ViT are pure Transformer models without convolution operation. The main difference between them is the pyramid structure.

> Our PVT breaks the routine of Transformer by introducing a progressive shrinking pyramid. It can generate multi-scale feature maps like a traditional CNN backbone.

> In addition, we also designed a simple but effective attention layer-SRA, to process high-resolution feature maps and reduce computation/memory costs.

> Benefiting from the above designs, our method has the following advantages over ViT:
> 1. more flexible - can generate feature maps of different scales, channels in different stages;
> 2. more versatile - can be easily plugged and played in most downstream task models;
> 3. more friendly to computation/memory - can process the feature map with higher resolution.

----------------------------------------------------------------------------------------------------

## References

## Further Reading

* [4] DETR
* [10] ViT
* [15] [ResNet](https://zhuanlan.zhihu.com/p/570072614)
* [16] SENet
* [18] DenseNet
* [22] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [24] SKNet
* [25] Generalized Focal Loss (GFL)
* [27] RetinaNet
* [30] SSD
* [41] [VGGNet](https://zhuanlan.zhihu.com/p/563314926)
* [46] [InceptionNetV1](https://zhuanlan.zhihu.com/p/564141144)
* [48] [EfficientNetV1](https://zhuanlan.zhihu.com/p/557565000)
* [51] Attention Is All You Need
* [56] ResNeXt
* [57] [Dilated Convolutions](https://zhuanlan.zhihu.com/p/555834549)
