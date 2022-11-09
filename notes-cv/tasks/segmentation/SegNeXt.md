# [Notes][Vision][Segmentation] SegNext <!-- omit in toc -->

* urls: [[abs](https://arxiv.org/abs/2209.08575)]
    [[pdf](https://arxiv.org/pdf/2209.08575.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2209.08575/)]
* Title: SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation
* Year: 18 Sep `2022`
* Authors: Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zhengning Liu, Ming-Ming Cheng, Shi-Min Hu
* Institutions: [Tsinghua University], [Nankai University], [Fitten Tech, Beijing, China]
* Abstract: We present SegNeXt, a simple convolutional network architecture for semantic segmentation. Recent transformer-based models have dominated the field of semantic segmentation due to the efficiency of self-attention in encoding spatial information. In this paper, we show that convolutional attention is a more efficient and effective way to encode contextual information than the self-attention mechanism in transformers. By re-examining the characteristics owned by successful segmentation models, we discover several key components leading to the performance improvement of segmentation models. This motivates us to design a novel convolutional attention network that uses cheap convolutional operations. Without bells and whistles, our SegNeXt significantly improves the performance of previous state-of-the-art methods on popular benchmarks, including ADE20K, Cityscapes, COCO-Stuff, Pascal VOC, Pascal Context, and iSAID. Notably, SegNeXt outperforms EfficientNet-L2 w/ NAS-FPN and achieves 90.6% mIoU on the Pascal VOC 2012 test leaderboard using only 1/10 parameters of it. On average, SegNeXt achieves about 2.0% mIoU improvements compared to the state-of-the-art methods on the ADE20K datasets with the same or fewer computations. Code is available at this https URL (Jittor) and this https URL (Pytorch).

## Table of Contents <!-- omit in toc -->

- [Summary of Main Contributions](#summary-of-main-contributions)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Method](#3-method)
  - [3.1 Convolutional Encoder](#31-convolutional-encoder)
  - [3.2 Decoder](#32-decoder)
- [4 Experiments](#4-experiments)
- [5 Conclusions and Discussion](#5-conclusions-and-discussion)
- [References](#references)
- [Further Reading](#further-reading)

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Proposed multi-scale convolutional attention (MSCA) module.
* 

----------------------------------------------------------------------------------------------------

## 1 Introduction

> Based on the above observation, we argue a successful semantic segmentation model should have the following characteristics:
> 1. A strong backbone network as encoder. Compared to previous CNN-based models, the performance improvement of transformer-based models is mostly from a stronger backbone network.
> 2. Multi-scale information interaction. Different from the image classification task that mostly identifies a single object, semantic segmentation is a dense prediction task and hence needs to process objects of varying sizes in a single image.
> 3. Spatial attention. Spatial attention allows models to perform segmentation through prioritization of areas within the semantic regions.
> 4. Low computational complexity. This is especially crucial when dealing with high-resolution images from remote sensing and urban scenes.

> Our network, termed SegNeXt, is mostly composed of convolutional operations except the decoder part, which contains a decomposition-based Hamburger module [21] (Ham) for global information extraction.

## 2 Related Work

## 3 Method

### 3.1 Convolutional Encoder

Notations:
* Let $F \in \mathbb{R}^{H \times W \times C}$ denote the input feature to the MSCA module.
* 

> The reasons why we choose depth-wise strip convolutions are two-fold.
> 1. strip convolution is lightweight.
> 2. there are some strip-like objects, such as human and telephone pole in the segmentation scenes.

> The down-sampling block has a convolution with stride 2 and kernel size 3 x 3, followed by a batch normalization layer [35].

> Note that, in each building block of MSCAN, we use batch normalization instead of layer normalization as we found batch normalization gains more for the segmentation performance.

### 3.2 Decoder

## 4 Experiments

## 5 Conclusions and Discussion

----------------------------------------------------------------------------------------------------

## References

* Guo, Meng-Hao, et al. "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation." *arXiv preprint arXiv:2209.08575* (2022).

## Further Reading

* [4] DeepLabV1
* [5] DeepLabV2
* [6] DeepLabV3
* [8] DeepLabV3+
* [27] ResNet
* [32] DenseNet
* [35] InceptionNetV2
* [53] Fully Convolutional Networks
* [81] ResNeXt

CNN-based Semantic Segmentation

* [1] SegNet
* [19] DANet
* [20] Res2net
* [45] GFF
* [64] U-Net
* [71] HRNet
* [86] Dilated Convolutions
* [87] OCRNet
* [94] PSPNet

Transformer-based Semantic Segmentation

* [10] Mask2Former
* [11] MaskFormer
* [44] Video K-Net
* [63] DPT
* [65] Segmenter
* [80] SegFormer
* [88] HRFormer
* [96] SETR

Capturing Global Context

* [23] EAMLP
* [34] CCNet
* [40] EMANet
* [89] OCNet
* [91] Context Encoding

Spatial Attentions

* [14] DeformableNetV1
* [17] ViT
* [22] PCT
* [51] SwinFormerV1

Channel Attention

* [9] SCA-CNN
* [30] SENet
* [72] ECA-Net

Recent Vision Transformers

* [33] FlowFormer
* [49] DSTT
* [50] FuseFormer
* [73] PVTV2
* [74] PVTV1
* [82] Focal Transformer
