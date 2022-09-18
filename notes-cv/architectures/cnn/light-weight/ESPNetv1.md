#! https://zhuanlan.zhihu.com/p/556122258
# [Notes][Vision][CNN] ESPNetv1

* url: https://arxiv.org/abs/1803.06815
* Title: ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
* Year: 19 Mar `2018`
* Authors: Sachin Mehta, Mohammad Rastegari, Anat Caspi, Linda Shapiro, Hannaneh Hajishirzi
* Abstract: We introduce a fast and efficient convolutional neural network, ESPNet, for semantic segmentation of high resolution images under resource constraints. ESPNet is based on a new convolutional module, efficient spatial pyramid (ESP), which is efficient in terms of computation, memory, and power. ESPNet is 22 times faster (on a standard GPU) and 180 times smaller than the state-of-the-art semantic segmentation network PSPNet, while its category-wise accuracy is only 8% less. We evaluated ESPNet on a variety of semantic segmentation datasets including Cityscapes, PASCAL VOC, and a breast biopsy whole slide image dataset. Under the same constraints on memory and computation, ESPNet outperforms all the current efficient CNN networks such as MobileNet, ShuffleNet, and ENet on both standard metrics and our newly introduced performance metrics that measure efficiency on edge devices. Our network can process high resolution images at a rate of 112 and 9 frames per second on a standard GPU and edge device, respectively.

----------------------------------------------------------------------------------------------------

## 1 Introduction

> ESP is based on a convolution factorization principle that decomposes a standard convolution into two steps: (1) point-wise convolutions and (2) spatial pyramid of dilated convolutions, as shown in Fig. 1. The point-wise convolutions help in reducing the computation, while the spatial pyramid of dilated convolutions re-samples the feature maps to learn the representations from large effective receptive field.

> We show that our ESP module is more efficient than other factorized forms of convolutions, such as Inception [11–13] and ResNext [14]. Under the same constraints on memory and computation, ESPNet outperforms MobileNet [16] and ShuffleNet [17] (two other efficient networks that are built upon the factorization principle).

> We note that existing spatial pyramid methods (e.g. the atrous spatial pyramid module in [3]) are computationally expensive and cannot be used at different spatial levels for learning the representations.

>  In contrast to these methods, ESP is computationally efficient and can be used at different spatial levels of a CNN network.

> Existing models based on dilated convolutions [1, 3, 18, 19] are large and inefficient, but our ESP module generalizes the use of dilated convolutions in a novel and efficient way.

## 2 Related Work

**Convolution factorization:**

> Convolutional factorization decomposes the convolutional operation into multiple steps to reduce the computational complexity. This factorization has successfully shown its potential in reducing the computational complexity of deep CNN networks (e.g. Inception [11–13], factorized network [22], ResNext [14], Xception [15], and MobileNets [16]).

**Network Compression:**

> Another approach for building efficient networks is compression. These methods use techniques such as hashing [23], pruning [24], vector quantization [25], and shrinking [26, 27] to reduce the size of the pre-trained network.

**Low-bit networks:**

> Another approach towards efficient networks is low-bit networks, which quantize the weights to reduce the network size and complexity (e.g. [28–31]).

**Sparse CNN:**

> To remove the redundancy in CNNs, sparse CNN methods, such as sparse decomposition [32], structural sparsity learning [33], and dictionary-based method [34], have been proposed.

**Dilated convolution:**

> Dilated convolutions [35] are a special form of standard convolutions in which the effective receptive field of kernels is increased by inserting zeros (or holes) between each pixel in the convolutional kernel.

**CNN for semantic segmentation:**

## 3 ESPNet

### 3.1 ESP module

> ESPNet is based on efficient spatial pyramid (ESP) modules, which are a factorized form of convolutions that decompose a standard convolution into a point-wise convolution and a spatial pyramid of dilated convolutions (see Fig. 1a).

> The point-wise convolution in the ESP module applies a 1x1 convolution to project high-dimensional feature maps onto a low-dimensional space.

> The spatial pyramid of dilated convolutions then re-samples these low-dimensional feature maps using $K$, $n \times n$ dilated convolutional kernels simultaneously, each with a dilation rate of $2^{k-1}$, $k = \{1, ..., K\}$.

> This factorization drastically reduces the number of parameters and the memory required by the ESP module, while preserving a large effective receptive field $[(n-1)2^{K-1} + 1]^{2}$. This pyramidal convolutional operation is called a spatial pyramid of dilated convolutions, because each dilated convolutional kernel learns weights with different receptive fields and so resembles a spatial pyramid.

Notations:
* Let $H, W \in \mathbb{Z}_{++}$ denote the height and width of the input feature map to the ESP module.
* Let $M \in \mathbb{Z}_{++}$ denote the number of input channels to the ESP module.
* Let $N \in \mathbb{Z}_{++}$ denote the number of output channels of the ESP module.
* Let $K \in \mathbb{Z}_{++}$ denote the width divider for the entire network.
* Let $*$ denote the convolution operation.
* Let $*_{l}$ denote the dilated convolution operation with dilation factor $l$.
* Let $x \in \mathbb{R}^{H \times W \times M}$ denote the input to the ESP module.
* Let $R^{(1)}, ..., R^{(\frac{N}{K})} \in \mathbb{R}^{1 \times 1 \times M}$ denote the point-wise convolutional kernels in the reduce step.
* Let $n \in \mathbb{Z}_{++}$ denote the height and width of the dilated convolutional kernels in the transform step.
* Let $T^{(1, 1)}, ..., T^{(K, \frac{N}{K})} \in \mathbb{R}^{n \times n \times \frac{N}{K}}$ denote the dilated convolutional kernels in the transform step.

The reduce step $reduce: \mathbb{R}^{H \times W \times M} \to \mathbb{R}^{H \times W \times \frac{N}{K}}$ is given by:
$$r := reduce(x) := \operatorname{Concat}(\{x * R^{(i)}: i \in \{1, ..., \frac{N}{K}\}\}).$$
The transform step $transform: \mathbb{R}^{H \times W \times \frac{N}{K}} \to \mathbb{R}^{H \times W \times \frac{N}{K}}$, for $k \in \{1, ..., K\}$ is given by:
$$t_{k} := transform(r) := \operatorname{Concat}(\{r *_{2^{k-1}} T^{(k, i)}: i \in \{1, ..., \frac{N}{K}\}\}).$$
The merge step $merge: [\mathbb{R}^{H \times W \times \frac{N}{K}}]^{K} \to \mathbb{R}^{H \times W \times N}$ is given by:
$$merge(t_{1}, ..., t_{K}) := \operatorname{Concat}(\{\sum_{i=1}^{k}t_{i}: k \in \{1, ..., K\}\}).$$

> To improve the gradient flow inside the network, the input and output feature maps of the ESP module are combined using an element-wise sum [47].

### 3.2 Relationship with other CNN modules

**MobileNet module**

> The MobileNet module [16], visualized in Fig. 3a, uses a depthwise separable convolution [15] that factorizes a standard convolutions into depth-wise convolutions (transform) and point-wise convolutions (expand).

> An extreme version of the ESP module (with $K = N$) is almost identical to the MobileNet module, differing only in the order of convolutional operations. In the MobileNet module, the spatial convolutions are followed by point-wise convolutions; however, in the ESP module, point-wise convolutions are followed by spatial convolutions.

> Note that the effective receptive field of an ESP module ($[(n−1)2^{K−1} + 1]^{2}$) is higher than a MobileNet module ($[n]^{2}$).

**ShuffleNet module**

> The ShuffleNet module [17], shown in Fig. 3b, is based on the principle of reduce-transform-expand. It is an optimized version of the bottleneck block in ResNet [47].

**Inception module**

> Inception modules [11–13] are built on the principle of split-reduce-transform-merge. These modules are usually heterogeneous in number of channels and kernel size (e.g. some of the modules are composed of standard and factored convolutions).

**ResNext module**

> A ResNext module [14], shown in Fig. 3d, is a parallel version of the bottleneck module in ResNet [47] and is based on the principle of split-reduce-transform-expand-merge.

**Atrous spatial pyramid (ASP) module**

> An ASP module [3], shown in Fig. 3e, is built on the principle of split-transform-merge. The ASP module involves branching with each branch learning kernel at a different receptive field (using dilated convolutions).

## 4 Experiments

### 4.1 Experimental set-up

> To build deeper computationally efficient networks for edge devices without changing the network topology, a hyper-parameter $\alpha$ controls the depth of the network; the ESP module is repeated $\alpha_{l}$ times at spatial level $l$.

### 4.2 Results on the Cityscape dataset

### 4.3 Segmentation results on other datasets

### 4.4 Performance analysis on an edge device

### 4.5 Ablation studies on the Cityscapes: The path from ESPNet-A to ESPNet

## 5 Conclusion

----------------------------------------------------------------------------------------------------

## References

* Mehta, Sachin, et al. "Espnet: Efficient spatial pyramid of dilated convolutions for semantic segmentation." *Proceedings of the european conference on computer vision (ECCV)*. 2018.

## Further Reading

* [1] PSPNet
* [3] DeepLabv2
* [11] [Inception-v1/GoogLeNet](https://zhuanlan.zhihu.com/p/564141144)
* [12] Inception-v3
* [13] Inception-v4
* [14] ResNeXt
* [15] [Xception Networks](https://zhuanlan.zhihu.com/p/556794897)
* [16] MobileNetV1
* [17] ShuffleNet V1
* [18] [Dilated Convolutions](https://zhuanlan.zhihu.com/p/555834549)
* [19] Dilated Residual Networks
* [22] Flattened Networks
* [47] ResNet
* [50] PReLU
* [ESPNetv2](https://zhuanlan.zhihu.com/p/556330935)
