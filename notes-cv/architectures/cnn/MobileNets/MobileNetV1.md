# [Notes][Vision][CNN] MobileNetV1

* url: https://arxiv.org/abs/1704.04861
* Title: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
* Year: 17 Apr `2017`
* Authors: Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
* Institutions: [Google Inc.]
* Abstract: We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* New model: MobileNet
* New ideas: width multiplier, resolution multiplier

----------------------------------------------------------------------------------------------------

## 1. Introduction

> This paper describes an efficient network architecture and a set of two hyper-parameters in order to build very small, low latency models that can be easily matched to the design requirements for mobile and embedded vision applications.

## 2. Prior Work



----------------------------------------------------------------------------------------------------

## Rerefences

* Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." *arXiv preprint arXiv:1704.04861* (2017).

## Further Reading

* [8] ResNet
* [16] Flattened Networks
* [19] AlexNet
* [27] VGGNet
* [29] Inception-v4/Inception-ResNet
* [31] Inception-v3
