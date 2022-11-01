# [Notes][Vision][CNN] MobileNetV1

* url: https://arxiv.org/abs/1704.04861
* Title: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
* Year: 17 Apr `2017`
* Authors: Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
* Institutions: [Google Inc.]
* Abstract: We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* New model architecture MobileNet based on depthwise separable convolutions.
* New ideas: width multiplier, resolution multiplier
* Improvement in not only size but also speed due to efficient GEMM.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> This paper describes an efficient network architecture and a set of two hyper-parameters in order to build very small, low latency models that can be easily matched to the design requirements for mobile and embedded vision applications.

## 2. Prior Work

> Many different approaches can be generally categorized into either compressing pretrained networks or training small networks directly.

> This paper proposes a class of network architectures that allows a model developer to specifically choose a small network that matches the resource restrictions (latency, size) for their application. MobileNets primarily focus on optimizing for latency but also yield small networks. Many papers on small networks focus only on size but do not consider speed.

## 3. MobileNet Architecture

### 3.1. Depthwise Separable Convolution

> The MobileNet model is based on depthwise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into a `depthwise convolution` and a 1x1 convolution called a `pointwise convolution`.

> A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a separate layer for filtering and a separate layer for combining.

> This factorization has the effect of drastically reducing computation and model size.

> First it uses depthwise separable convolutions to break the interaction between the number of output channels and the size of the kernel.

> The standard convolution operation has the effect of filtering features based on the convolutional kernels and combining features in order to produce a new representation. The filtering and combination steps can be split into two steps via the use of factorized convolutions called depthwise separable convolutions for substantial reduction in computational cost.

> MobileNets use both batchnorm and ReLU nonlinearities for both layers.

> Additional factorization in spatial dimension such as in [16, 31] does not save much additional computation as very little computation is spent in depthwise convolutions.

Notations:
* Let $M \in \mathbb{Z}_{++}$ denote the number of input channels.
* Let $N \in \mathbb{Z}_{++}$ denote the number of output channels.
* Let $F_{H}, F_{W} \in \mathbb{Z}_{++}$ denote the height and width of the input feature map.
* Let $K_{H}, K_{W} \in \mathbb{Z}_{++}$ denote the height and width of the kernel.

Then the number of parameters in the three layers are:

<center>

| Layer | Parameters |
|-------|------------|
| Standard | $K_{H}K_{W}F_{H}F_{W}MN$ |
| Depthwise | $K_{H}K_{W}F_{H}F_{W}M$ |
| Pointwise | $F_{H}F_{W}MN$ |

</center>

The ratio between the number of parameters of a separable convolution and a standard convolution is:
$$\frac{\text{separable}}{\text{standard}} = \frac{K_{H}K_{W}F_{H}F_{W}M + F_{H}F_{W}MN}{K_{H}K_{W}F_{H}F_{W}MN} = \frac{1}{N} + \frac{1}{K_{H}K_{W}} << 1.$$

The ratio between the number of parameters of a depthwise convolution and a pointwise convolution is:
$$\frac{\text{depthwise}}{\text{pointwise}} = \frac{K_{H}K_{W}F_{H}F_{W}M}{F_{H}F_{W}MN} = \frac{K_{H}K_{W}}{N} << 1.$$

### 3.2. Network Structure and Training

> All layers are followed by a batchnorm [13] and ReLU nonlinearity with the exception of the final fully connected layer which has no nonlinearity and feeds into a softmax layer for classification.

> Down sampling is handled with strided convolution in the depthwise convolutions as well as in the first layer.

> A final average pooling reduces the spatial resolution to 1 before the fully connected layer.

Optimizing for Speed

> Our model structure puts nearly all of the computation into dense 1x1 convolutions. This can be implemented with highly optimized general matrix multiply (GEMM) functions.

Training Methodologies

> Contrary to training large models we use less regularization and data augmentation techniques because small models have less trouble with overfitting.

> Additionally, we found that it was important to put very little or no weight decay (l2 regularization) on the depthwise filters since their are so few parameters in them.

### 3.3. Width Multiplier: Thinner Models

> The role of the width multiplier $\alpha$ is to thin a network uniformly at each layer. For a given layer and width multiplier $\alpha$, the number of input channels $M$ becomes $\alpha M$ and the number of output channels $N$ becomes $\alpha N$.

Notations:
* Let $\alpha \in (0, 1]$ denote the width multiplier.

Then the number of parameters in different kinds of layers are:

<center>

| Layer | Parameters |
|-------|------------|
| Standard, $\alpha$ | $K_{H}K_{W}F_{H}F_{W}\alpha M\alpha N$ |
| Depthwise, $\alpha$ | $K_{H}K_{W}F_{H}F_{W}\alpha M$ |
| Pointwise, $\alpha$ | $F_{H}F_{W}\alpha M\alpha N$ |

</center>

> Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly $\alpha^{2}$. Width multiplier can be applied to any model structure to define a new smaller model with a reasonable accuracy, latency and size trade off. It is used to define a new reduced structure that needs to be trained from scratch.

### 3.4. Resolution Multiplier: Reduced Representation

Notations:
* Let $\rho \in (0, 1]$ denote the resolution multiplier.

Then the number of parameters in different kinds of layers are:

<center>

| Layer | Parameters |
|-------|------------|
| Standard, $\alpha$, $\rho$ | $K_{H}K_{W}\rho F_{H}\rho F_{W}\alpha M\alpha N$ |
| Depthwise, $\alpha$, $\rho$ | $K_{H}K_{W}\rho F_{H}\rho F_{W}\alpha M$ |
| Pointwise, $\alpha$, $\rho$ | $\rho F_{H}\rho F_{W}\alpha M\alpha N$ |

</center>

> Resolution multiplier has the effect of reducing computational cost by $\rho^{2}$.

## 4. Experiments

## 5. Conclusion

----------------------------------------------------------------------------------------------------

## Rerefences

* Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." *arXiv preprint arXiv:1704.04861* (2017).

## Further Reading

* [3] [XceptionNet](https://zhuanlan.zhihu.com/p/556794897)
* [8] [ResNet](https://zhuanlan.zhihu.com/p/570072614)
* [12] SqueezeNet
* [13] Inception-v2/Batch Normalization
* [16] Flattened Networks
* [19] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [21] SSD
* [23] Faster R-CNN
* [27] [VGGNet](https://zhuanlan.zhihu.com/p/563314926)
* [29] Inception-v4/Inception-ResNet
* [30] [InceptionNetV1](https://zhuanlan.zhihu.com/p/564141144)
* [31] Inception-v3
* [34] Factorized Networks
