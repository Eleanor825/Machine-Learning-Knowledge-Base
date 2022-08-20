#! https://zhuanlan.zhihu.com/p/555834549
# [Notes][Vision][CNN] Dilated Convolutions

* url: https://arxiv.org/abs/1511.07122
* Title: Multi-Scale Context Aggregation by Dilated Convolutions
* Year: 23 Nov `2015`
* Authors: Fisher Yu, Vladlen Koltun
* Abstract: State-of-the-art models for semantic segmentation are based on adaptations of convolutional networks that had originally been designed for image classification. However, dense prediction and image classification are structurally different. In this work, we develop a new convolutional network module that is specifically designed for dense prediction. The presented module uses dilated convolutions to systematically aggregate multi-scale contextual information without losing resolution. The architecture is based on the fact that dilated convolutions support exponential expansion of the receptive field without loss of resolution or coverage. We show that the presented context module increases the accuracy of state-of-the-art semantic segmentation systems. In addition, we examine the adaptation of image classification networks to dense prediction and show that simplifying the adapted network can increase accuracy.

----------------------------------------------------------------------------------------------------

## 1 Introduction

> In this work, we develop a convolutional network module that aggregates multi-scale contextual information without losing resolution or analyzing rescaled images. The module can be plugged into existing architectures at any resolution.

> Unlike pyramid-shaped architectures carried over from image classification, the presented context module is designed specifically for dense prediction. It is a rectangular prism of convolutional layers, with no pooling or subsampling. The module is based on dilated convolutions, which support exponential expansion of the receptive field without loss of resolution or coverage.

## 2 Dilated Convolutions

Notations:
* Let $H, W \in \mathbb{Z}$ denote the height and width of the input feature map.
* Let $\mathcal{D} \subset \mathbb{Z}^{2}$ be given by $\mathcal{D} := ([0, H-1] \oplus [0, W-1]) \cap \mathbb{Z}^{2}$.
* Let $F: \mathcal{D} \to \mathbb{R}$ denote the input feature map.
* Let $\tilde{F}: \mathbb{Z}^{2} \to \mathbb{R}$ given by $\tilde{F}(p) := F(p)$ if $p \in \mathcal{D}$ and $\tilde{F}(p) := 0$ if $p \notin \mathcal{D}$ denote the padded input feature map.
* Let $\Omega_{r} \subseteq \mathbb{Z}^{2}$ be given by $\Omega_{r} := [-r, +r]^{2} \cap \mathbb{Z}^{2}$.
* Let $k: \Omega_{r} \to \mathbb{R}$ denote the convolutional kernel of size $(2r+1) \times (2r+1)$.
* Let $l \in \mathbb{Z}_{++}$ denote the dilation factor.

Then the standard convolution $F * k: \mathcal{D} \to \mathbb{R}$ can be formulated as
$$(F * k)(p) := \sum_{t \in \Omega_{r}}\tilde{F}(p - t)k(t) \tag{1}$$
and the dilated convolution $F *_{l} k: \mathcal{D} \to \mathbb{R}$ with dilation factor $l$ is given by
$$(F *_{l} k)(p) := \sum_{t \in \Omega_{r}}\tilde{F}(p - l \cdot t)k(t). \tag{2}$$

> We use the term "dilated convolution" instead of "convolution with a dilated filter" to clarify that no “dilated filter” is constructed or represented. The convolution operator itself is modified to use the filter parameters in a different way. The dilated convolution operator can apply the same filter at different ranges using different dilation factors. Our definition reflects the proper implementation of the dilated convolution operator, which does not involve construction of dilated filters.

Notations:
* Let $F_{0}: \mathbb{Z}^{2} \to \mathbb{R}$ denote the input feature map.
* Let $k_{0}, ..., k_{n-2}: \Omega_{1} \to \mathbb{R}$ denote discrete $3 \times 3$ filters.
* Define feature maps $F_{1}, ..., F_{n-1}$ by $F_{i+1} := F_{i} *_{2^{i}} k_{i}$ for $i = 0, ..., n-2$.

> It is easy to see that the size of the receptive field of each element in $F_{i+1}$ is $(2^{i+2}-1) \times (2^{i+2} - 1)$. The receptive field is a square of exponentially increasing size.

## 3 Multi-Scale Context Aggregation



----------------------------------------------------------------------------------------------------

## References

* Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions." *arXiv preprint arXiv:1511.07122* (2015).

## Further Reading

