# [Notes][Vision][CNN] MobileNetV2

* url: https://arxiv.org/abs/1801.04381
* Title: MobileNetV2: Inverted Residuals and Linear Bottlenecks
* Year: 13 Jan `2018`
* Authors: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
* Institutions: [Google Inc.]
* Abstract: In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3. The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Network design is based on MobileNetV1.
* New module: inverted residual with linear bottleneck.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> Our network pushes the state of the art for mobile tailored computer vision models, by significantly decreasing the number of operations and memory needed while retaining the same accuracy.

> Our main contribution is a novel layer module: the inverted residual with linear bottleneck. This module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear convolution.

## 2. Related Work

> Our network design is based on MobileNetV1 [26]. It retains its simplicity and significantly improves its accuracy, achieving state of the art on multiple image classification and detection tasks for mobile applications.

## 3. Preliminaries, discussion and intuition

### 3.1 Depthwise Separable Convolutions

> The basic idea is to replace a full convolutional operator with a factorized version that splits convolution into two separate layers.
> 1. The first layer is called a depthwise convolution, it performs lightweight filtering by applying a single convolutional filter per input channel.
> 2. The second layer is a 1x1 convolution, called a pointwise convolution, which is responsible for building new features through computing linear combinations of the input channels.

> Effectively depthwise separable convolution reduces computation compared to traditional layers by almost a factor of $k^{2}$. MobileNetV2 uses $k=3$ (3x3 depthwise separable convolutions) so the computational cost is 8 to 9 times smaller than that of standard convolutions at only a small reduction in accuracy [26].

### 3.2 Linear Bottlenecks

Consider a deep neural network.

Notations:
* Let $n \in \mathbb{Z}_{++}$ denote the number of layers in the network.
* Let $L_{1}, ..., L_{n}$ denote the layers in the network.
* Let $h_{1}, w_{i}, d_{i} \in \mathbb{Z}_{++}$ denote the dimensions of the activation tensor of layer $L_{i}$, for $i \in \{1, ..., n\}$.

> Informally, for an input set of real images, we say that the set of layer activations (for any layer $L_{i}$) forms a "manifold of interest".

> It has been long assumed that manifolds of interest in neural networks could be embedded in low-dimensional subspaces. In other words, when we look at all individual $d$-channel pixels of a deep convolutional layer, the information encoded in those values actually lie in some manifold, which in turn is embeddable into a low-dimensional subspace.

> To summarize, we have highlighted two properties that are indicative of the requirement that the manifold of interest should lie in a low-dimensional subspace of the higher-dimensional activation space:
> 1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
> 2. ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.

> These two insights provide us with an empirical hint for optimizing existing neural architectures: assuming the manifold of interest is low-dimensional we can capture this by inserting linear bottleneck layers into the convolutional blocks. Experimental evidence suggests that using linear layers is crucial as it prevents non-linearities from destroying too much information.

> We will refer to the ratio between the size of the input bottleneck and the inner size as the `expansion ratio`.

### 3.3. Inverted residuals

> Inspired by the intuition that the bottlenecks actually contain all the necessary information, while an expansion layer acts merely as an implementation detail that accompanies a non-linear transformation of the tensor, we use shortcuts directly between the bottlenecks.

> The motivation for inserting shortcuts is similar to that of classical residual connections: we want to improve the ability of a gradient to propagate across multiplier layers.

> The crucial difference, however, is that in residual networks the bottleneck layers are treated as low-dimensional supplements to high-dimensional “information” tensors.

Notations:
* Let $H, W \in \mathbb{Z}_{++}$ denote the height and width of the input feature map to a module.
* Let $k \in \mathbb{Z}_{++}$ denote the kernel size.
* Let $M \in \mathbb{Z}+{++}$ denote the number of input channels.
* Let $N \in \mathbb{Z}_{++}$ denote the number of output channels.
* Let $t \in \mathbb{Z}_{++}$ denote the expansion factor.

Then the number of multiply-add operations in different modules are:

<center>

| Module | Operations |
|--------|------------|
| Depthwise Convolutional Layer | $k^{2}MHW + MHWN = HWM(k^{2} + N)$ |
| Inverted Residual Block | $MHWtM + k^{2}tMHW + tMHWN = HWMt(M + k^{2} + N)$ |

</center>

### 3.4. Information flow interpretation

> One interesting property of our architecture is that it provides a natural separation between the input/output `domains` of the building blocks (bottleneck layers), and the layer `transformation` - that is a non-linear function that converts input to the output. The former can be seen as the `capacity` of the network at each layer, whereas the latter as the `expressiveness`.

> This is in contrast with traditional convolutional blocks, both regular and separable, where both expressiveness and capacity are tangled together and are functions of the output layer depth.

Different choices of the expansion ratio:
* > When inner layer depth is 0 the underlying convolution is the identity function thanks to the shortcut connection.
* > When the expansion ratio is smaller than 1, this is a classical residual convolutional block [1, 28].
* > However, for our purposes we show that expansion ratio greater than 1 is the most useful.

## 4. Model Architecture

> We use ReLU6 as the non-linearity because of its robustness when used with low-precision computation [26].

> We always use kernel size 3x3 as is standard for modern networks, and utilize dropout and batch normalization during training.

## 5. Implementation Notes

## 6. Experiments

### 6.4. Ablation study

**Inverted residual connections**

> The new result reported in this paper is that the shortcut connecting bottleneck perform better than shortcuts connecting the expanded layers (see Figure 5(b) for comparison).

**Importance of linear bottlenecks**

> Theoretically, the linear bottleneck models are strictly less powerful than models with non-linearities, because the activations can always operate in linear regime with appropriate changes to biases and scaling.

> However our experiments shown in Figure 5(a) indicate that linear bottlenecks improve performance, providing a strong support for the hypothesis that a non-linearity operator is not beneficial in the low-dimensional space of a bottleneck.

## 7. Conclusions and future work

> On the theoretical side: the proposed convolutional block has a unique property that allows to separate the network expressivity (encoded by expansion layers) from its capacity (encoded by bottleneck inputs). Exploring this is an important direction for future research.

----------------------------------------------------------------------------------------------------

## References

## Further Reading

* [1] ResNet
* [5] AlexNet
* [6] VGGNet
* [7] [Inception-v1/GoogLeNet](https://zhuanlan.zhihu.com/p/564141144)
* [19] ShuffleNet
* [22] NASNet
* [26] MobileNetV1
* [27] [Xception Networks](https://zhuanlan.zhihu.com/p/556794897)
* [28] ResNeXt
