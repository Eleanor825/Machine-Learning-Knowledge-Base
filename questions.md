## PVT
* why 4 8 16 32?
* dimensions of the matrices in PVT
* what does "append learnable classification token" mean
* review FPN
* what is pre-trained position embedding
<!-- * dilated convolution [74] -->
<!-- * NAS [61] -->
<!-- * Res2Net [17] -->
<!-- * ResNeSt [79] -->

## others
* what is the difference between Deep Roots and ResNeXt???
* 用convolution去掉格子

## EfficientNetV1
* input/output space of the convolutions? where do $H_{i}$, $W_{i}$, and $C_{i}$ change?

## Inception-v2/Batch Normalization

* The Inception Networks used depthwise separable convolution?

## Inception-v4/Inception-ResNet

* Uniform simplified architecture of Inception-v4?

## MobileNetV2

* 3.4 information flow interpretation?

## DeepMask
* in segmentation loss, what is the range of i and j? Do we use the full ground truth segmentation mask or a crop of size $w^{o} \times h^{o}$ of it?
* how's low-rank achieved? SVD?

## R-CNN
* Did R-CNN refine spatial locations? Fast R-CNN seemed to say yes.
* It has a third training stage? to train the bounding box regressors?

## For CS 484 Prof.

* Is it a good thing to train a fully convolutional network on a dataset that contains images of various sizes, without preprocessing to unify them?
* Point processing using activation from different channels but at the same location? Does pointwise convolution count as point processing?
* What is $L$ in those diagrams?
