# [MobileNetV1](https://arxiv.org/abs/1704.04861)

* Title: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
* Year: 17 Apr `2017`
* Author: Andrew G. Howard
* Abstract: We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.

----------------------------------------------------------------------------------------------------

* Depthwise Separable Convolution: \
    Separated convolution operation into a depth-wise convolution and a point-wise convolution to reduce computations.
