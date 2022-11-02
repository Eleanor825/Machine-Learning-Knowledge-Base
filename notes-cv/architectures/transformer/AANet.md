#! https://zhuanlan.zhihu.com/p/579585272
# [Notes][Vision][Transformer] AANet <!-- omit in toc -->

* urls: [[abs](https://arxiv.org/abs/1904.09925)]
    [[pdf](https://arxiv.org/pdf/1904.09925.pdf)]
    [vanity]
* Title: Attention Augmented Convolutional Networks
* Year: 22 Apr `2019`
* Authors: Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, Quoc V. Le
* Abstract: Convolutional networks have been the paradigm of choice in many computer vision applications. The convolution operation however has a significant weakness in that it only operates on a local neighborhood, thus missing global information. Self-attention, on the other hand, has emerged as a recent advance to capture long range interactions, but has mostly been applied to sequence modeling and generative modeling tasks. In this paper, we consider the use of self-attention for discriminative visual tasks as an alternative to convolutions. We introduce a novel two-dimensional relative self-attention mechanism that proves competitive in replacing convolutions as a stand-alone computational primitive for image classification. We find in control experiments that the best results are obtained when combining both convolutions and self-attention. We therefore propose to augment convolutional operators with this self-attention mechanism by concatenating convolutional feature maps with a set of feature maps produced via self-attention. Extensive experiments show that Attention Augmentation leads to consistent improvements in image classification on ImageNet and object detection on COCO across many different models and scales, including ResNets and a state-of-the art mobile constrained network, while keeping the number of parameters similar. In particular, our method achieves a $1.3\%$ top-1 accuracy improvement on ImageNet classification over a ResNet50 baseline and outperforms other attention mechanisms for images such as Squeeze-and-Excitation. It also achieves an improvement of 1.4 mAP in COCO Object Detection on top of a RetinaNet baseline.

## Table of Contents <!-- omit in toc -->

- [Summary of Main Contributions](#summary-of-main-contributions)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
  - [2.2. Attention mechanisms in networks](#22-attention-mechanisms-in-networks)
- [3. Methods](#3-methods)
  - [3.1. Self-attention over images](#31-self-attention-over-images)
      - [3.1.1 Two-dimensional Positional Embeddings](#311-two-dimensional-positional-embeddings)
  - [3.2. Attention Augmented Convolution](#32-attention-augmented-convolution)
- [4. Experiments](#4-experiments)
- [5. Discussion and future work](#5-discussion-and-future-work)
- [References](#references)
- [Further Reading](#further-reading)

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Proposed relative positional embeddings.
* Proposed to augment convolutions with multi-head self-attentions.

----------------------------------------------------------------------------------------------------

## 1. Introduction

>  The design of the convolutional layer imposes 1) locality via a limited receptive field and 2) translation equivariance via weight sharing. Both these properties prove to be crucial inductive biases when designing models that operate over images. However, the local nature of the convolutional kernel prevents it from capturing global contexts in an image, often necessary for better recognition of objects in images [33].

> Surprisingly, experiments also reveal that fully self-attentional models, a special case of Attention Augmentation, only perform slightly worse than their fully convolutional counterparts on ImageNet, indicating that self-attention is a powerful standalone `computational primitive` for image classification.

## 2. Related Work

### 2.2. Attention mechanisms in networks

> In contrast, our attention augmented networks do not rely on pretraining of their fully convolutional counterparts and employ self-attention along the entire architecture. The use of multi-head attention allows the model to attend jointly to both spatial and feature subspaces.

> Additionally, we enhance the representational power of self-attention over images by extending relative self-attention [37, 18] to two dimensional inputs allowing us to model translation equivariance in a principled way.

> Finally our method produces additional feature maps, rather than recalibrating convolutional features via addition [45, 53] or gating [17, 16, 31, 46]. This property allows us to flexibly adjust the fraction of attentional channels and consider a spectrum of architectures, ranging from fully convolutional to fully attentional models.

## 3. Methods

### 3.1. Self-attention over images

Notations:
* Let $H, W \in \mathbb{Z}_{++}$ denote the height and width of the input image.
* Let $F_{in} \in \mathbb{Z}_{++}$ denote the number of channels in the input image.
* Let $X \in \mathbb{R}^{HW \times F_{in}}$ denote the flattened version of the input image.
* Let $d_{k} \in \mathbb{Z}_{++}$ denote the query depth and the key depth.
* Let $d_{v} \in \mathbb{Z}_{++}$ denote the value depth (attentional channels).
* Let $N_{h} \in \mathbb{Z}_{++}$ denote the number of heads.
* Let $W_{q}, W_{k} \in \mathbb{R}^{F_{in} \times d_{k}^{h}}$ denote the matrix representation of the query and key transforms.
* Let $W_{v} \in \mathbb{R}^{F_{in} \times d_{v}^{h}}$ denote the matrix representation of the value transform.
* Let $O_{h} \in \mathbb{R}^{HW \times d_{v}^{h}}$ denote the output tensor of the $h$-th head, for $h \in \{1, ..., N_{h}\}$.
* Let $W^{O} \in \mathbb{R}^{d_{v} \times d_{v}}$ denote the output projection matrix.
* Let $\operatorname{MHA}: \mathbb{R}^{HW \times F_{in}} \to \mathbb{R}^{HW \times d_{v}}$ denote the multi-head self-attention operation.

Then
$$O_{h} := \operatorname{Softmax}\bigg(\frac{(XW_{q})(XW_{k})^{\top}}{\sqrt{d_{k}^{h}}}\bigg)(XW_{v}) \text{ and }$$
$$\operatorname{MHA}(X) := \operatorname{Concatenate}[O_{1}, ..., O_{N_{h}}]W^{O}.$$

##### 3.1.1 Two-dimensional Positional Embeddings

> Without explicit information about positions, self-attention is permutation equivariant:
> $$\operatorname{MHA}(\pi(X)) = \pi(\operatorname{MHA}(X))$$
> for any permutation π of the pixel locations, making it ineffective for modeling highly structured data such as images.

**Relative positional embeddings**

Notations:
* Let $Q = XW_{q} \in \mathbb{R}^{HW \times d_{k}^{h}}$ denote the query matrix.
* Let $K = XW_{k} \in \mathbb{R}^{HW \times d_{k}^{h}}$ denote the key matrix.
* Let $V = XW_{v} \in \mathbb{R}^{HW \times d_{v}^{h}}$ denote the value matrix.
* Let $S_{H}^{rel}, S_{W}^{rel} \in \mathbb{R}^{HW \times HW}$ denote the matrices of relative position logits along height and width dimensions.
* Let $r^{H}_{j_{y}-i_{y}} \in \mathbb{R}^{d_{k}^{h}}$ denote the learned embedding for relative height.
* Let $r^{W}_{j_{x}-i_{x}} \in \mathbb{R}^{d_{k}^{h}}$ denote the learned embedding for relative width.

Then
$$[S_{H}^{rel}]_{ij} := q_{i}^{\top}r^{H}_{j_{y}-i_{y}},
[S_{W}^{rel}]_{ij} := q_{i}^{\top}r^{W}_{j_{x}-i_{x}}, \text{ and }$$
$$O_{h} := \operatorname{Softmax}\bigg(\frac{QK^{\top} + S_{H}^{rel} + S_{W}^{rel}}{\sqrt{d_{k}^{h}}}\bigg)V.$$

> The relative positional embeddings $r^{W}$ and $r^{H}$ are learned and shared across heads but not layers. For each layer, we add $(2(H+W)-2)d_{k}^{h}$ parameters to model relative distances along height and width.

### 3.2. Attention Augmented Convolution

> In contrast to these approaches, we
> 1) use an attention mechanism that can attend jointly to spatial and feature subspaces (each head corresponding to a feature subspace) and
> 2) introduce additional feature maps rather than refining them.

**Concatenating convolutional and attentional feature maps**

Notations:
* Let $F_{out} \in \mathbb{Z}_{++}$ denote the number of output channels of $\operatorname{Conv}$.
* Let $\operatorname{Conv}: \mathbb{R}^{HW \times F_{in}} \to \mathbb{R}^{HW \times F_{out}}$ denote the convolution operation.

We define the $\operatorname{AAConv}: \mathbb{R}^{HW \times F_{in}} \to \mathbb{R}^{HW \times (F_{out}+d_{v})}$ by
$$\operatorname{AAConv}(X) := \operatorname{Concatenate}[\operatorname{Conv}(X), \operatorname{MHA}(X)].$$

> Similarly to the convolution, the proposed attention augmented convolution
> 1) is equivariant to translation and
> 2) an readily operate on inputs of different spatial dimensions.

**Effect on number of parameters**

**Attention Augmented Convolutional Architectures**

> Since the memory cost $O(N_{h}(HW)^{2})$ can be prohibitive for large spatial dimensions, we augment convolutions with attention starting from the last layer (with smallest spatial dimension) until we hit memory constraints.

## 4. Experiments

## 5. Discussion and future work

----------------------------------------------------------------------------------------------------

## References

* Bello, Irwan, et al. "Attention augmented convolutional networks." *Proceedings of the IEEE/CVF international conference on computer vision*. 2019.

## Further Reading

* [13] [ResNet](https://zhuanlan.zhihu.com/p/570072614)
* [14] ResNetV2
* [17] SENet
* [20] InceptionNetV2
* [23] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [24] LeNet
* [36] MobileNetV2
* [39] [InceptionNetV4](https://zhuanlan.zhihu.com/p/568801341)
* [40] [InceptionNetV1](https://zhuanlan.zhihu.com/p/564141144)
* [41] InceptionNetV3
* [42] MnasNet
* [43] [Transformer](https://zhuanlan.zhihu.com/p/566630656)
* [47] ResNeXt
