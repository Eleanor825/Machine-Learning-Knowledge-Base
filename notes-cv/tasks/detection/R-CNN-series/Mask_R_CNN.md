# [Notes][Vision][Detection] Mask R-CNN <!-- omit in toc -->

* urls: [[abs](https://arxiv.org/abs/1703.06870)]
    [[pdf](https://arxiv.org/pdf/1703.06870.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.06870/)]
* Title: Mask R-CNN
* Year: 20 Mar `2017`
* Authors: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
* Abstract: We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code has been made available at: this https URL

## Table of Contents <!-- omit in toc -->
- [Summary of Main Contributions](#summary-of-main-contributions)
- [1. Introduction](#1-introduction)
- [References](#references)
- [Further Reading](#further-reading)

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Proposed RoIAlign layer.
* Decoupled mask and class prediction.
* 

----------------------------------------------------------------------------------------------------

## 1. Introduction

> Instance segmentation is challenging because it requires the correct detection of all objects in an image while also precisely segmenting each instance. It therefore combines elements from the classical computer vision tasks of object detection, where the goal is to classify individual objects and localize each using a bounding box, and semantic segmentation, where the goal is to classify each pixel into a fixed set of categories without differentiating object instances.
>
> 

----------------------------------------------------------------------------------------------------

## References

* He, Kaiming, et al. "Mask r-cnn." *Proceedings of the IEEE international conference on computer vision*. 2017.

## Further Reading

* [9] Fast R-CNN
* [10] R-CNN
* [13] Spatial Pyramid Pooling Networks (SPPNet)
* [18] AlexNet
* [21] Feature Pyramid Networks (FPN)
* [23] Fully Convolutional Networks (FCN)
* [26] DeepMask
* [27] SharpMask
* [28] Faster R-CNN
* [29] OHEM
* 