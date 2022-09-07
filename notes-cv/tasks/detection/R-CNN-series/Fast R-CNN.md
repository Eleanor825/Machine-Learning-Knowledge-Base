# [Notes][Vision][Detection] Fast R-CNN

* url: https://arxiv.org/abs/1504.08083
* Title: Fast R-CNN
* Year: 30 Apr `2015`
* Authors: Ross Girshick
* Institutions: [Microsoft Research]
* Abstract: This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3x faster, tests 10x faster, and is more accurate. Fast R-CNN is implemented in Python and C++ (using Caffe) and is available under the open-source MIT License at [this https URL](https://github.com/rbgirshick/fast-rcnn).

----------------------------------------------------------------------------------------------------

## 1. Introduction

> In this paper, we streamline the training process for state-of-the-art ConvNet-based object detectors [9, 11]. We propose a single-stage training algorithm that jointly learns to classify object proposals and refine their spatial locations.

### 1.1. R-CNN and SPPnet

R-CNN

> R-CNN, however, has notable drawbacks:
> 1. **Training is a multi-stage pipeline**. R-CNN first fine-tunes a ConvNet on object proposals using log loss. Then, it fits SVMs to ConvNet features. These SVMs act as object detectors, replacing the softmax classifier learnt by fine-tuning. In the third training stage, bounding-box regressors are learned.
> 2. **Training is expensive in space and time**. For SVM and bounding-box regressor training, features are extracted from each object proposal in each image and written to disk.
> 3. **Object detection is slow**. At test-time, features are extracted from each object proposal in each test image.

> R-CNN is slow because it performs a ConvNet forward pass for each object proposal, without sharing computation.

Spatial Pyramid Pooling

> Spatial pyramid pooling networks (SPPnets) [11] were proposed to speed up R-CNN by sharing computation.

> The SPPnet method computes a convolutional feature map for the entire input image and then classifies each object proposal using a feature vector extracted from the shared feature map.

----------------------------------------------------------------------------------------------------

## References

* Girshick, Ross. "Fast r-cnn." *Proceedings of the IEEE international conference on computer vision*. 2015.

## Further Reading

* [9] R-CNN
* [11] Spatial Pyramid Pooling (SPP)
* [14] AlexNet
* [15] Spatial Pyramid Matching
* [19] OverFeat
* [25] segDeepM
