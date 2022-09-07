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

Drawbacks of R-CNN

> R-CNN, however, has notable drawbacks:
> 1. **Training is a multi-stage pipeline**. R-CNN first fine-tunes a ConvNet on object proposals using log loss. Then, it fits SVMs to ConvNet features. These SVMs act as object detectors, replacing the softmax classifier learnt by fine-tuning. In the third training stage, bounding-box regressors are learned.
> 2. **Training is expensive in space and time**. For SVM and bounding-box regressor training, features are extracted from each object proposal in each image and written to disk.
> 3. **Object detection is slow**. At test-time, features are extracted from each object proposal in each test image.

> R-CNN is slow because it performs a ConvNet forward pass for each object proposal, without sharing computation.

Advantages of Spatial Pyramid Pooling

> Spatial pyramid pooling networks (SPPnets) [11] were proposed to speed up R-CNN by sharing computation.

> The SPPnet method computes a convolutional feature map for the entire input image and then classifies each object proposal using a feature vector extracted from the shared feature map.

> Features are extracted for a proposal by max-pooling the portion of the feature map inside the proposal into a fixed-size output (\eg, 6x6). Multiple output sizes are pooled and then concatenated as in spatial pyramid pooling [15].

Drawbacks of Spatial Pyramid Pooling

> SPPnet also has notable drawbacks. Like R-CNN, training is a multi-stage pipeline that involves extracting features, fine-tuning a network with log loss, training SVMs, and finally fitting bounding-box regressors. Features are also written to disk.

> But unlike R-CNN, the fine-tuning algorithm proposed in [11] cannot update the convolutional layers that precede the spatial pyramid pooling.

### 1.2. Contributions

Advantages of Fast R-CNN

> The Fast R-CNN method has several advantages:
> 1. Higher detection quality (mAP) than R-CNN, SPPnet
> 2. Training is single-stage, using a multi-task loss
> 3. Training can update all network layers
> 4. No disk storage is required for feature caching

## 2. Fast R-CNN architecture and training

> 1. A Fast R-CNN network takes as input an entire image and a set of `object proposals`.
> 2. The network first processes the whole image with several convolutional (conv) and max pooling layers to produce a conv `feature map`.
> 3. Then, for each object proposal a region of interest (RoI) pooling layer extracts a fixed-length `feature vector` from the feature map.
> 4. Each feature vector is fed into a sequence of fully connected (fc) layers that finally branch into two sibling output layers:
> 5. one that produces softmax probability estimates over K object classes plus a catch-all "background" class
> 6. and another layer that outputs four real-valued numbers for each of the K object classes. Each set of 4 values encodes refined bounding-box positions for one of the K classes.

### 2.1. The RoI pooling layer

> The RoI pooling layer uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of $H \times W$ (\eg, 7x7), where H and W are layer hyper-parameters that are independent of any particular RoI.

> The RoI layer is simply the special-case of the spatial pyramid pooling layer used in SPPnets [11] in which there is only one pyramid level.

### 2.2. Initializing from pre-trained networks

> When a pre-trained network initializes a Fast R-CNN network, it undergoes three transformations.
> 1. First, the last max pooling layer is replaced by a RoI pooling layer that is configured by setting $H$ and $W$ to be compatible with the net's first fully connected layer (\eg, $H=W=7$ for VGG16).
> 2. Second, the network's last fully connected layer and softmax (which were trained for 1000-way ImageNet classification) are replaced with the two sibling layers described earlier (a fully connected layer and softmax over $K+1$ categories and category-specific bounding-box regressors).
> 3. Third, the network is modified to take two data inputs: a list of images and a list of RoIs in those images.

### 2.3. Fine-tuning for detection



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
