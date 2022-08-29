# [Notes][Vision][Segmentation] ERFNet

* url: https://ieeexplore.ieee.org/document/8063438
* Title: ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation
* Year: 09 October `2017`
* Authors: Eduardo Romera; José M. Álvarez; Luis M. Bergasa; Roberto Arroyo
* Institutions: [Department of Electronics, University of Alcalá, Alcalá de Henares, Spain], [CSIRO-Data61, Canberra, Australia]
* Abstract: Semantic segmentation is a challenging task that addresses most of the perception needs of intelligent vehicles (IVs) in an unified way. Deep neural networks excel at this task, as they can be trained end-to-end to accurately classify multiple object categories in an image at pixel level. However, a good tradeoff between high quality and computational resources is yet not present in the state-of-the-art semantic segmentation approaches, limiting their application in real vehicles. In this paper, we propose a deep architecture that is able to run in real time while providing accurate semantic segmentation. The core of our architecture is a novel layer that uses residual connections and factorized convolutions in order to remain efficient while retaining remarkable accuracy. Our approach is able to run at over 83 FPS in a single Titan X, and 7 FPS in a Jetson TX1 (embedded device). A comprehensive set of experiments on the publicly available Cityscapes data set demonstrates that our system achieves an accuracy that is similar to the state of the art, while being orders of magnitude faster to compute than other architectures that achieve top precision. The resulting tradeoff makes our model an ideal approach for scene understanding in IV applications. The code is publicly available at: https://github.com/Eromera/erfnet.

----------------------------------------------------------------------------------------------------

## I. INTRODUCTION

> The core element of our architecture is a novel layer design that leverages skip connections and convolutions with 1D kernels.

> While the skip connections allow the convolutions to learn residual functions that facilitate training, the 1D factorized convolutions allow a significant reduction of the computational costs while retaining a similar accuracy compared to the 2D ones.

## II. RELATED WORKS

## III. ERFNET: PROPOSED ARCHITECTURE

### A. Factorized Residual Layers

> We propose to redesign the non-bottleneck residual module in a more optimal way by entirely using convolutions with 1D filters (Fig. 2 (c)).

### B. Architecture Design

## IV. EXPERIMENTS

### A. General Setup

## V. CONCLUSION AND FUTURE WORKS

----------------------------------------------------------------------------------------------------

## References

* Romera, Eduardo, et al. "Erfnet: Efficient residual factorized convnet for real-time semantic segmentation." *IEEE Transactions on Intelligent Transportation Systems* 19.1 (2017): 263-272.

## Further Reading

* [4] AlexNet
* [5] Fully Convolutional Networks (FCN)
* [6] ResNet
* [7] Wide Residual Networks (WRN)
* [8] DeepLabv2
* [9] RefineNet
* [11] ENet
* [15] VGG
* [16] SegNet-Basic
* [17] DeepLabv1
* [19] Laplacian Pyramid Reconstruction and Refinement (LRR)
* [25] PReLU
