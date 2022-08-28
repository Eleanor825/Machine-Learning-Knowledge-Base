# [Notes][Vision][Segmentation] Laplacian Pyramid Reconstruction and Refinement

* url: https://arxiv.org/abs/1605.02264
* Title: Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation
* Year: 08 May `2016`
* Authors: Golnaz Ghiasi, Charless C. Fowlkes
* Institutions: [Dept. of Computer Science, University of California, Irvine]
* Abstract: CNN architectures have terrific recognition performance but rely on spatial pooling which makes it difficult to adapt them to tasks that require dense, pixel-accurate labeling. This paper makes two contributions: (1) We demonstrate that while the apparent spatial resolution of convolutional feature maps is low, the high-dimensional feature representation contains significant sub-pixel localization information. (2) We describe a multi-resolution reconstruction architecture based on a Laplacian pyramid that uses skip connections from higher resolution feature maps and multiplicative gating to successively refine segment boundaries reconstructed from lower-resolution maps. This approach yields state-of-the-art semantic segmentation results on the PASCAL VOC and Cityscapes segmentation benchmarks without resorting to more complex random-field inference or instance detection driven architectures.

----------------------------------------------------------------------------------------------------

## 1 Introduction

> Inspired in part by recent work on residual networks [16, 17], we propose an architecture in which predictions derived from high-resolution layers are only required to correct residual errors in the low-resolution prediction. Importantly, we use multiplicative gating to avoid integrating (and hence penalizing) noisy high-resolution outputs in regions where the low-resolution predictions are confident about the semantic content.

## 2 Related Work

> One insight is that spatial information lost during max-pooling can in part be recovered by unpooling and deconvolution [36] providing a useful way to visualize input dependency in feed-forward models [35].

> A second key insight is that while activation maps at lower-levels of the CNN hierarchy lack object category specificity, they do contain higher spatial resolution information.

## 3 Reconstruction with learned basis functions

**Reconstruction by deconvolution**

**Connection to spline interpolation**

**Learning basis functions**

## 4 Laplacian Pyramid Refinement

----------------------------------------------------------------------------------------------------

## References

* Ghiasi, Golnaz, and Charless C. Fowlkes. "Laplacian pyramid reconstruction and refinement for semantic segmentation." *European conference on computer vision*. Springer, Cham, 2016.

## Further Reading

* [15] Hypercolumns
* [16] ResNet
* [17] Identity Mappings
* [24] Fully Convolutional Networks (FCN)
* [26] Learning Deconvolution
* [36] Adaptive Deconvolutional Networks
