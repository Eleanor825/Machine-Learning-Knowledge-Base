# [Notes][Vision][Segmentation] ERFNet

* urls: [[abs](https://ieeexplore.ieee.org/document/8063438)]
        [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8063438)]
* Title: ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation
* Year: 09 Oct `2017`
* Authors: Eduardo Romera; José M. Álvarez; Luis M. Bergasa; Roberto Arroyo
* Institutions: [Department of Electronics, University of Alcalá, Alcalá de Henares, Spain], [CSIRO-Data61, Canberra, Australia]
* Abstract: Semantic segmentation is a challenging task that addresses most of the perception needs of intelligent vehicles (IVs) in an unified way. Deep neural networks excel at this task, as they can be trained end-to-end to accurately classify multiple object categories in an image at pixel level. However, a good tradeoff between high quality and computational resources is yet not present in the state-of-the-art semantic segmentation approaches, limiting their application in real vehicles. In this paper, we propose a deep architecture that is able to run in real time while providing accurate semantic segmentation. The core of our architecture is a novel layer that uses residual connections and factorized convolutions in order to remain efficient while retaining remarkable accuracy. Our approach is able to run at over 83 FPS in a single Titan X, and 7 FPS in a Jetson TX1 (embedded device). A comprehensive set of experiments on the publicly available Cityscapes data set demonstrates that our system achieves an accuracy that is similar to the state of the art, while being orders of magnitude faster to compute than other architectures that achieve top precision. The resulting tradeoff makes our model an ideal approach for scene understanding in IV applications. The code is publicly available at: https://github.com/Eromera/erfnet.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> The core element of our architecture is a novel layer design that leverages skip connections and convolutions with 1D kernels.

> While the skip connections allow the convolutions to learn residual functions that facilitate training, the 1D factorized convolutions allow a significant reduction of the computational costs while retaining a similar accuracy compared to the 2D ones.

## 2. Related Works

## 3. ERFNet: Proposed Architecture

### 3.1. Factorized Residual Layers

> We propose to redesign the non-bottleneck residual module in a more optimal way by entirely using convolutions with 1D filters (Fig. 2 (c)).

> This module is faster (as in computation time) and has less parameters than the bottleneck design, while keeping a learning capacity and accuracy equivalent to the non-bottleneck one.

> Although the focus of this paper is the segmentation task, the proposed non-bottleneck-1D design is directly transferable to any existing network that makes use of residual layers, including both classification and segmentation architectures.

**Increased Network Width**

> Additionally, this design facilitates a direct increase in the "width" (seen as the number of filters or of feature maps computed), while keeping at a minimum the computational resources. Increasing width has already been proven effective in classification-aimed residual networks [7], [21].

> The segmentation architecture proposed in the next section demonstrates that dense prediction tasks like semantic segmentation can also benefit from the increased width, while remaining computationally efficient due to the proposed 1D factorization.

### 3.2. Architecture Design

## 4. Experiments

### 4.1. General Setup

## 5. Conclusion and Future Works

----------------------------------------------------------------------------------------------------

## References

* Romera, Eduardo, et al. "Erfnet: Efficient residual factorized convnet for real-time semantic segmentation." *IEEE Transactions on Intelligent Transportation Systems* 19.1 (2017): 263-272.

## See Also

* [4] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [5] [Fully Convolutional Networks (FCN)](https://zhuanlan.zhihu.com/p/561031110)
* [6] [ResNet](https://zhuanlan.zhihu.com/p/570072614)
* [7] Wide Residual Networks (WRN)
* [8] DeepLabV2
* [9] [RefineNet](https://zhuanlan.zhihu.com/p/577767167)
* [11] ENet
* [15] [VGGNet](https://zhuanlan.zhihu.com/p/563314926)
* [16] SegNet-Basic
* [17] DeepLabV1
* [19] Laplacian Pyramid Reconstruction and Refinement (LRR)
* [25] PReLU
