#! https://zhuanlan.zhihu.com/p/556330935
# [Notes][Vision][CNN] ESPNetv2

* url: https://arxiv.org/abs/1811.11431
* Title: ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network
* Year: 28 Nov `2018`
* Authors: Sachin Mehta, Mohammad Rastegari, Linda Shapiro, Hannaneh Hajishirzi
* Abstract: We introduce a light-weight, power efficient, and general purpose convolutional neural network, ESPNetv2, for modeling visual and sequential data. Our network uses group point-wise and depth-wise dilated separable convolutions to learn representations from a large effective receptive field with fewer FLOPs and parameters. The performance of our network is evaluated on four different tasks: (1) object classification, (2) semantic segmentation, (3) object detection, and (4) language modeling. Experiments on these tasks, including image classification on the ImageNet and language modeling on the PenTree bank dataset, demonstrate the superior performance of our method over the state-of-the-art methods. Our network outperforms ESPNet by 4-5% and has 2-4x fewer FLOPs on the PASCAL VOC and the Cityscapes dataset. Compared to YOLOv2 on the MS-COCO object detection, ESPNetv2 delivers 4.4% higher accuracy with 6x fewer FLOPs. Our experiments show that ESPNetv2 is much more power efficient than existing state-of-the-art efficient methods including ShuffleNets and MobileNets. Our code is open-source and available at [this https URL](https://github.com/sacmehta/ESPNetv2).

----------------------------------------------------------------------------------------------------

## 1. Introduction

Main contributions:
* > (1) A general purpose architecture for modeling both visual and sequential data efficiently.
* > (2) Our proposed architecture, ESPNetv2, extends ESPNet [32], a dilated convolution-based segmentation network, with depth-wise separable convolutions; an efficient form of convolutions that are used in state-of-art efficient networks including MobileNets [17, 44] and ShuffleNets [29, 60]. 
* > (3) Our empirical results show that ESPNetv2 delivers similar or better performance with fewer FLOPS on different visual recognition tasks.

> We also study a cyclic learning rate scheduler with warm restarts. Our results suggests that this scheduler is more effective than the standard fixed learning rate scheduler.

## 2. Related Work

**Efficient CNN architectures**

> Most state-of-the-art efficient networks [17, 29, 44] use depth-wise separable convolutions [17] that factor a convolution into two steps to reduce computational complexity.

>  Another efficient form of convolution that has been used in efficient networks [18,60] is group convolution [22].

> In addition to convolutional factorization, a network’s efficiency and accuracy can be further improved using methods such as channel shuffle [29] and channel split [29].

**Neural architecture search**

> These approaches search over a huge network space using a pre-defined dictionary containing different parameters, including different convolutional layers, different convolutional units, and different filter sizes [4, 52, 56, 66]. Recent search-based methods [52, 56] have shown improvements for MobileNetv2. 

**Network compression**

**Low-bit representation**

## 3. ESPNetv2

### 3.1. Depth-wise dilated separable convolution

Notations:
* Let $n \in \mathbb{Z}_{++}$ denote the height and width of the convolutional kernel.
* Let $c \in \mathbb{Z}_{++}$ denote the number of input channels of the layer.
* Let $\hat{c} \in \mathbb{Z}_{++}$ denote the number of output channels of the layer.
* Let $g \in \mathbb{Z}_{++}$ denote the number of filter groups of the grouped convolutional layer.
* Let $r \in \mathbb{Z}_{++}$ denote the dilation factor of the dilated convolutional layer.
* Define $n_{r} := (n-1) \cdot r + 1$.

| Convolution Type             | Parameters          | Effective Receptive Field |
| ---------------------------- | ------------------- | ------------------------- |
| Standard                     | $n^{2}c\hat{c}$     | $n \times n$              |
| Group                        | $n^{2}c\hat{c}/g$   | $n \times n$              |
| Depth-wise separable         | $n^{2}c + c\hat{c}$ | $n \times n$              |
| Depth-wise dilated separable | $n^{2}c + c\hat{c}$ | $n_{r} \times n_{r}$      |

### 3.2. EESP unit

> To make the ESP module even more computationally efficient, we first replace point-wise convolutions with group point-wise convolutions.

> We then replace computationally expensive 3 x 3 dilated convolutions with their economical counterparts i.e. depth-wise dilated separable convolutions.

> To remove the gridding artifacts caused by dilated convolutions, we fuse the feature maps using the computationally efficient hierarchical feature fusion (HFF) method [32].

**Strided EESP with shortcut connection to an input image**

> To learn representations efficiently at multiple scales, we make following changes to the EESP block in Figure 1c:
> 1. depth-wise dilated convolutions are replaced with their strided counterpart,
> 2. an average pooling operation is added instead of an identity connection,
> 3. the element-wise addition operation is replaced with a concatenation operation, which helps in expanding the dimensions of feature maps efficiently [60].

> Spatial information is lost during down-sampling and convolution (filtering) operations. To better encode spatial relationships and learn representations efficiently, we add an efficient long-range shortcut connection between the input image and the current down-sampling unit. This connection first down-samples the image to the same size as that of the feature map and then learns the representations using a stack of two convolutions. The first convolution is a standard 3 x 3 convolution that learns the spatial representations while the second convolution is a point-wise convolution that learns linear combinations between the input, and projects it to a high-dimensional space.

### 3.3. Network architecture

> In our experiments, we set the dilation rate r proportional to the number of branches in the EESP unit ($K$). The effective receptive field of the EESP unit grows with $K$. Some of the kernels, especially at low spatial levels such as 7 x 7, might have a larger effective receptive field than the size of the feature map. Therefore, such kernels might not contribute to learning. In order to have meaningful kernels, we limit the effective receptive field at each spatial level $l$ with spatial dimension $W^{l} \times H^{l}$ as:
$$n_{d}^{l}(Z^{l}) := 5 + \frac{Z^{l}}{7}, \quad Z^{l} \in \{H^{l}, W^{l}\}$$
> with the effective receptive field $(n_{d} \times n_{d})$ corresponding to the lowest spatial level (i.e. 7x7) as 5x5.

## 4. Experiments

## 5. Ablation Studies on the ImageNet Dataset

## 6. Conclusion

----------------------------------------------------------------------------------------------------

## References

* Mehta, Sachin, et al. "Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.

## Further Reading

* [5] DeepLabv2
* [15] PReLU
* [16] [ResNet](https://zhuanlan.zhihu.com/p/570072614)
* [17] MobileNetV1
* [18] CondenseNet
* [21] Inception-v2/Batch Normalization
* [22] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [29] ShuffleNet V2
* [32] [ESPNetv1](https://zhuanlan.zhihu.com/p/556122258)
* [44] MobileNetV2
* [51] [InceptionNetV1/GoogLeNet](https://zhuanlan.zhihu.com/p/564141144)
* [52] MnasNet
* [60] ShuffleNet V1
* [63] PSPNet
