# [Notes][Vision][CNN] Dilated Convolutions

* url: https://arxiv.org/abs/1511.07122
* Title: Multi-Scale Context Aggregation by Dilated Convolutions
* Year: 23 Nov `2015`
* Authors: Fisher Yu, Vladlen Koltun
* Abstract: State-of-the-art models for semantic segmentation are based on adaptations of convolutional networks that had originally been designed for image classification. However, dense prediction and image classification are structurally different. In this work, we develop a new convolutional network module that is specifically designed for dense prediction. The presented module uses dilated convolutions to systematically aggregate multi-scale contextual information without losing resolution. The architecture is based on the fact that dilated convolutions support exponential expansion of the receptive field without loss of resolution or coverage. We show that the presented context module increases the accuracy of state-of-the-art semantic segmentation systems. In addition, we examine the adaptation of image classification networks to dense prediction and show that simplifying the adapted network can increase accuracy.

----------------------------------------------------------------------------------------------------

## 1 Introduction

> In this work, we develop a convolutional network module that aggregates multi-scale contextual information without losing resolution or analyzing rescaled images. The module can be plugged
into existing architectures at any resolution.

> Unlike pyramid-shaped architectures carried over from image classification, the presented context module is designed specifically for dense prediction. It is a rectangular prism of convolutional layers, with no pooling or subsampling. 

----------------------------------------------------------------------------------------------------

## References

## Further Reading

