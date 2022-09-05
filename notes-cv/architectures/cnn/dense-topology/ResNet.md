# [Notes][Vision][CNN] ResNet

* url: https://arxiv.org/abs/1512.03385
* Title: Deep Residual Learning for Image Recognition
* Year: 10 Dec `2015`
* Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
* Institutions: [Microsoft Research]
* Abstract: Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

----------------------------------------------------------------------------------------------------

## 1. Introduction

The Problem with Deep Networks

> When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly.

> Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [11, 42] and thoroughly verified by our experiments.

> The degradation (of training accuracy) indicates that not all systems are similarly easy to optimize.

> Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it. There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart. But experiments show that our current solvers on hand are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time).

Our Approach

> Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping.

> Formally, denoting the desired underlying mapping as $\mathcal{H}(x)$, we let the stacked nonlinear layers fit another mapping of $\mathcal{F}(x) := \mathcal{H}(x) - x$. The original mapping is recast into $\mathcal{F}(x) + x$. We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

<p align="center">
    <img src="ResNet_figure_2.png">
</p>

> We show that:
> 1. Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets (that simply stack layers) exhibit higher training error when the depth increases;
> 2. Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks.

## 2. Related Work

**Residual Representations**



## Identity vs. Projection Shortcuts

Three Options for Increasing Dimensions:
* Zero-padding shortcuts are used for increasing dimensions. All shortcuts are parameter-free.
* Projection shortcuts are used for increasing dimensions. Other shortcuts are identity.
* All shortcuts are projection shortcuts.

All three options performed similarly.
So don't use the third one.

## Bottleneck Architecture

* Each residual block consists of 3 convolutional layers.
    * 1st layer: $1 \times 1$. Responsible for decreasing dimensions.
    * 2nd layer: $3 \times 3$.
    * 3rd layer: $1 \times 1$. Responsible for increasing dimensions.

The parameter-free identity shortcuts are particularly important for the bottleneck architectures.
If the identity shortcut is replaced with projection, one can show that the time complexity and model size are doubled,
as the shortcut is connected to the two high-dimensional ends.

----------------------------------------------------------------------------------------------------

## References

* He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

## Further Reading

* [7] Fast R-CNN
* [8] R-CNN
* [12] Spatial Pyramid Pooling (SPP)
* [16] Inception-v2/Batch Normalization
* [21] AlexNet
* [27] [Fully Convolutional Networks (FCN)](https://zhuanlan.zhihu.com/p/561031110)
* [32] Faster R-CNN
* [40] OverFeat
* [41] VGG
* [42] [Highway Networks](https://zhuanlan.zhihu.com/p/554615809)
* [44] Inception-v1/GoogLeNet
