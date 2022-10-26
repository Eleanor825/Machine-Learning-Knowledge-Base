# [Notes][Vision][Segmentation] RefineNet <!-- omit in toc -->

* url: https://arxiv.org/abs/1611.06612
* Title: RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
* Year: 20 Nov `2016`
* Authors: Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid
* Institutions: [The University of Adelaide], [Australian Centre for Robotic Vision]
* Abstract: Recently, very deep convolutional neural networks (CNNs) have shown outstanding performance in object recognition and have also been the first choice for dense classification problems such as semantic segmentation. However, repeated subsampling operations like pooling or convolution striding in deep CNNs lead to a significant decrease in the initial image resolution. Here, we present RefineNet, a generic multi-path refinement network that explicitly exploits all the information available along the down-sampling process to enable high-resolution prediction using long-range residual connections. In this way, the deeper layers that capture high-level semantic features can be directly refined using fine-grained features from earlier convolutions. The individual components of RefineNet employ residual connections following the identity mapping mindset, which allows for effective end-to-end training. Further, we introduce chained residual pooling, which captures rich background context in an efficient manner. We carry out comprehensive experiments and set new state-of-the-art results on seven public datasets. In particular, we achieve an intersection-over-union score of 83.4 on the challenging PASCAL VOC 2012 dataset, which is the best reported result to date.

## Table of Contents <!-- omit in toc -->

- [Summary of Main Contributions](#summary-of-main-contributions)
- [1. Introduction](#1-introduction)
  - [1.1. Related Work](#11-related-work)
- [References](#references)
- [Further Reading](#further-reading)

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Proposed RefineNet.
* Proposed Chained Residual Pooling.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> Multiple stages of spatial pooling and convolution strides reduce the final image prediction typically by a factor of 32 in each dimension, thereby losing much of the finer image structure.

> One way to address this limitation is to learn deconvolutional filters as an up-sampling operation [38, 36] to generate high-resolution feature maps. The deconvolution operations are not able to recover the low-level visual features which are lost after the down-sampling operation in the convolution forward stage. Therefore, they are unable to output accurate high-resolution prediction. Low-level visual information is essential for accurate prediction on the boundaries or details.

> Another type of methods exploits features from intermediate layers for generating high-resolution prediction, e.g., the FCN method in [36] and Hypercolumns in [22]. The intuition behind these works is that features from middle layers are expected to describe mid-level representations for object parts, while retaining spatial information. This information is though to be complementary to the features from early convolution layers which encode low-level spatial visual information like edges, corners, circles, etc., and also complementary to high-level features from deeper layers which encode high-level semantic information, including object- or category-level evidence, but which lack strong spatial information.

> We argue that features from all levels are helpful for semantic segmentation. High-level semantic features helps the category recognition of image regions, while low-level visual features help to generate sharp, detailed boundaries for high-resolution prediction.

### 1.1. Related Work



----------------------------------------------------------------------------------------------------

## References

## Further Reading

* [2] SegNet
* [5] DeepLabV1
* [18] R-CNN
* [22] Hypercolumn
* [24] ResNet
* [25] ResNetV2
* [36] Fully Convolutional Networks (FCN)
* [38] DeconvNet
* [42] VGGNet
* 