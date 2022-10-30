#! https://zhuanlan.zhihu.com/p/567335485
# [Notes][Vision][Transformer] Swin Transformer V1

* url: https://arxiv.org/abs/2103.14030
* Title: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
* Year: 25 Mar `2021`
* Authors: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
* Institutions: [Microsoft Research Asia]
* Abstract: This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \textbf{S}hifted \textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at [this https URL](https://github.com/microsoft/Swin-Transformer).

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Proposed Swin Transformer, a general-purpose Transformer backbone.
* Linear complexity w.r.t. input image size.
* Key design: shift of the window partition between consecutive self-attention layers.
* Replaced the standard Multi-Head Self-Attention (MSA) with shifted-window based MSA.
* Techniques used: GELU activation, layer normalization, cyclic-shifting, relative position bias.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> In this paper, we seek to expand the applicability of Transformer such that it can serve as a general-purpose backbone for computer vision, as it does for NLP and as CNNs do in vision.

> We observe that significant challenges in transferring its high performance in the language domain to the visual domain can be explained by differences between the two modalities.
> 1. One of these differences involves scale. Unlike the word tokens that serve as the basic elements of processing in language Transformers, visual elements can vary substantially in scale, a problem that receives attention in tasks such as object detection [41, 52, 53]. In existing Transformer-based models [61, 19], tokens are all of a fixed scale, a property unsuitable for these vision applications.
> 2. Another difference is the much higher resolution of pixels in images compared to words in passages of text. There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high-resolution images, as the computational complexity of its self-attention is quadratic to image size.

> To overcome these issues, we propose a general-purpose Transformer backbone, called Swin Transformer, which constructs `hierarchical feature maps` and has `linear computational complexity` to image size.

> These merits make Swin Transformer suitable as a general-purpose backbone for various vision tasks, in contrast to previous Transformer based architectures [19] which produce feature maps of a single resolution and have quadratic complexity.

> A key design element of Swin Transformer is its shift of the window partition between consecutive self-attention layers, as illustrated in Figure 2.

> Our experiments show that the proposed shifted window approach has much lower `latency` than the sliding window method, yet is similar in modeling power (see Tables 5 and 6).

> It is our belief that a unified architecture across computer vision and natural language processing could benefit both fields, since it would facilitate joint modeling of visual and textual signals and the modeling knowledge from both domains can be more deeply shared.

## 2. Related Work

## 3. Method

<figure align="center">
    <img src="Swin_Transformer_V1_figure_3">
    <figcaption> Figure 3: (a) The architecture of a Swin Transformer (Swin-T); (b) two successive Swin Transformer Blocks (notation presented with Eq. (3)). W-MSA and SW-MSA are multi-head self attention modules with regular and shifted windowing configurations, respectively. </figcaption>
</figure>

### 3.1. Overall Architecture

**Patch Splitting**

> It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT. Each patch is treated as a "token" and its feature is set as a concatenation of the raw pixel RGB values.

**Linear Embedding**

> A `linear embedding` layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as C).

**Stage 1**

> Several Transformer blocks with modified self-attention computation (`Swin Transformer blocks`) are applied on these patch tokens. The Transformer blocks maintain the number of tokens ($\frac{H}{4} \times \frac{W}{4}$), and together with the linear embedding
are referred to as "Stage 1".

**Stage 2, Stage 3, and Stage 4**

> To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper.

**Patch Merging**

> The first `patch merging` layer concatenates the features of each group of 2x2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of 2x2 = 4 (2x downsampling of resolution), and the output dimension is set to 2C.

> Swin Transformer blocks are applied afterwards for feature transformation, with the resolution kept at $\frac{H}{8} \times \frac{W}{8}$.

> The procedure is repeated twice, as "Stage 3" and "Stage 4", with output resolutions of $\frac{H}{16} \times \frac{W}{16}$ and $\frac{H}{32} \times \frac{W}{32}$, respectively.

> These stages jointly produce a hierarchical representation, with the same feature map resolutions as those of typical convolutional networks, e.g., VGG [51] and ResNet [29]. As a result, the proposed architecture can conveniently replace the backbone networks in existing methods for various vision tasks.

### 3.2 Shifted Window based Self-Attention

**Self-attention in non-overlapped windows**

Notations:
* Let $C \in \mathbb{Z}_{++}$ denote the dimension of the patch embedding.
* Let $h, w \in \mathbb{Z}_{++}$ denote the number patches per col/row.
* Let $M \in \mathbb{Z}_{++}$ denote the number of patches in a winder per row/col.

Then the computational complexity of MSA and W-MSA are:

<center>

| Module | Complexity |
|--------|------------|
| MSA | $4hwC^{2} + 2(hw)^{2}C$ |
| W-MSA | $4hwC^{2} + 2M^{2}hwC$ |

</center>

**Shifted window partitioning in successive blocks**

> The window-based self-attention module lacks connections across windows, which limits its modeling power. To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, we propose a `shifted window partitioning` approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks.

$$\begin{align*}
    \hat{\textbf{z}}^{l} & = \operatorname{W-MSA}(\operatorname{LN}(\textbf{z}^{l-1})) + \textbf{z}^{l-1}, \\
    \textbf{z}^{l} & = \operatorname{MLP}(\operatorname{LN}(\hat{\textbf{z}}^{l})) + \hat{\textbf{z}}^{l}, \\
    \hat{\textbf{z}}^{l+1} & = \operatorname{SW-MSA}(\operatorname{LN}(\textbf{z}^{l})) + \textbf{z}^{l}, \\
    \textbf{z}^{l+1} & = \operatorname{MLP}(\operatorname{LN}(\hat{\textbf{z}}^{l+1})) + \hat{\textbf{z}}^{l+1}.
    \end{align*}$$
where $\hat{\textbf{z}}^{l}$ and $\textbf{z}^{l}$ denote the output features of the (S)W-MSA module and the MLP module for block $l$, respectively; W-MSA and SW-MSA denote window based multi-head self-attention using regular and shifted window partitioning configurations, respectively.

> The shifted window partitioning approach introduces connections between neighboring non-overlapping windows in the previous layer and is found to be effective in image classification, object detection, and semantic segmentation.

**Efficient batch computation for shifted configuration**

> Here, we propose a more efficient batch computation approach by cyclic-shifting toward the top-left direction, as illustrated in Figure 4.

> After this shift, a batched window may be composed of several sub-windows that are not adjacent in the feature map, so a masking mechanism is employed to limit self-attention computation to within each sub-window.

> With the cyclic-shift, the number of batched windows remains the same as that of regular window partitioning, and thus is also efficient.

**Relative position bias**

Notations:
* Let $M \in \mathbb{Z}_{++}$ denote the number of patches in a winder per row/col.
* Let $Q, K, V \in \mathbb{R}^{M^{2} \times d}$ denote the query, key, and value matrices.
* Let $B \in \mathbb{R}^{M^{2} \times M^{2}}$ denote the relative position bias.

$$\operatorname{Attention}(Q, K, V) = \operatorname{SoftMax}(\frac{QK^{\top}}{\sqrt{d}} + B)V$$

> We observe significant improvements over counterparts without this bias term or that use absolute position embedding, as shown in Table 4.

### 3.3. Architecture Variants

## 4. Experiments

### 4.4. Ablation Study

**Shifted windows**

**Relative position bias**

**Different self-attention methods**

> The self-attention modules built on the proposed shifted window approach are 40.8x/2.5x, 20.2x/2.5x, 9.3x/2.1x, and 7.6x/1.8x more efficient than those of sliding windows in naive/kernel implementations on four network stages, respectively.

> Overall, the Swin Transformer architectures built on shifted windows are 4.1/1.5, 4.0/1.5, 3.6/1.5 times faster than variants built on sliding windows for Swin-T, Swin-S, and Swin-B, respectively.

## 5. Conclusion

> This paper presents Swin Transformer, a new vision Transformer which produces a hierarchical feature representation and has linear computational complexity with respect to input image size.

> As a key element of Swin Transformer, the shifted window based self-attention is shown to be effective and efficient on vision problems, and we look forward to investigating its use in natural language processing as well.

----------------------------------------------------------------------------------------------------

## References

* Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021.

## Further Reading

* [17] Deformable Convnets v1
* [19] [Vision Transformer (ViT)](https://zhuanlan.zhihu.com/p/566567383)
* [29] [ResNet](https://zhuanlan.zhihu.com/p/570072614)
* [33] DenseNet
* [38] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [41] Feature Pyramid Networks (FPN)
* [50] U-Net
* [51] [VGGNet](https://zhuanlan.zhihu.com/p/563314926)
* [56] [Inception-v1/GoogLeNet](https://zhuanlan.zhihu.com/p/564141144)
* [57] [EfficientNetV1](https://zhuanlan.zhihu.com/p/557565000)
* [60] DeiT
* [61] [Transformer](https://zhuanlan.zhihu.com/p/566630656)
* [67] ResNeXt
* [73] Wide Residual Networks (WRN)
* [78] SETR
* [81] Deformable Convnets v2
