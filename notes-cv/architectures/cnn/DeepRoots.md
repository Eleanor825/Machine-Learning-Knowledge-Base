#! https://zhuanlan.zhihu.com/p/555413761
# [Notes][Vision][CNN] Deep Roots

* url: https://arxiv.org/abs/1605.06489
* Title: Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups
* Year: 20 May `2016`
* Authors: Yani Ioannou, Duncan Robertson, Roberto Cipolla, Antonio Criminisi
* Abstract: We propose a new method for creating computationally efficient and compact convolutional neural networks (CNNs) using a novel sparse connection structure that resembles a tree root. This allows a significant reduction in computational cost and number of parameters compared to state-of-the-art deep CNNs, without compromising accuracy, by exploiting the sparsity of inter-layer filter dependencies. We validate our approach by using it to train more efficient variants of state-of-the-art CNN architectures, evaluated on the CIFAR10 and ILSVRC datasets. Our results show similar or higher accuracy than the baseline architectures with much less computation, as measured by CPU and GPU timings. For example, for ResNet 50, our model has 40% fewer parameters, 45% fewer floating point operations, and is 31% (12%) faster on a CPU (GPU). For the deeper ResNet 200 our model has 25% fewer floating point operations and 44% fewer parameters, while maintaining state-of-the-art accuracy. For GoogLeNet, our model has 7% fewer parameters and is 21% (16%) faster on a CPU (GPU).

----------------------------------------------------------------------------------------------------

## 1. Introduction

> In this work we will show that a similar idea can be applied to the channel extents - i.e. filter inter-connectivity - by using filter groups [4].

## 2. Related Work

> Most previous work on reducing the computational complexity of CNNs has focused on approximating convolutional filters in the spatial (as opposed to the channel) domain, either by using low-rank approximations [9–13], or Fourier transform based convolution [14, 15].

> More general methods have used reduced precision number representations [16] or compression of previously trained models [17, 18].

> Here we explore methods that reduce the computational impact of the large number of filter channels within state-of-the art networks. Specifically, we consider decreasing the number of incoming connections to nodes.

**AlexNet Filter Groups**

> The authors observe that independent filter groups learn a separation of responsibility (colour features vs. texture features) that is consistent over different random initializations.

> Also surprising, and not explicitly stated in [4], is the fact that the AlexNet network has approximately 57% fewer connection weights than the corresponding network without filter groups. This is due to the reduction in the input channel dimension of the grouped convolution filters (see Fig. 2).

**Low-Dimensional Embeddings**

> Lin et al. [19] proposed a method to reduce the dimensionality of convolutional feature maps. By using relatively cheap '1x1' convolutional layers (i.e. layers comprising $d$ filters of size $1 \times 1 \times c$, where $d < c$), they learn to map feature maps into lower dimensional spaces, i.e. to new feature maps with fewer channels. Subsequent spatial filters operating on this lower dimensional input space require significantly less computation.

> This method is used in most state of the art networks for image classification to reduce computation [2, 20]. Our method is complementary.

**GoogLeNet**

> GoogLeNet uses, as a basic building block, a mixture of low-dimensional embeddings [19] and heterogeneously sized spatial filters - collectively an 'inception' module.

>  GoogLeNet is by far the most efficient state-of-the-art network for ILSVRC, achieving near state-of-the-art accuracy with the lowest computation/model size. However, we will show that even such an efficient and optimized network architecture benefits from our method.

**Low-Rank Approximations**

> Various authors have suggested approximating learned convolutional filters using tensor decomposition [11, 13, 18].

> In this paper we are not approximating an existing model’s weights but creating a new network architecture with explicit structural sparsity, which is then trained from scratch.

**Learning a Basis for Filters**

> Our approach is connected with that of Ioannou et al. [9] who showed that replacing $3 \times 3 \times c$ filters with linear combinations of filters with smaller spatial extent (e.g. $1 \times 3 \times c$, $3 \times 1 \times c$ filters, see Fig. 3) could reduce the model size and computational complexity of state-of-the-art CNNs, while maintaining or even increasing accuracy. However, that work did not address the channel extent of the filters.

## 3. Root Architectures

**Learning a Basis for Filter Dependencies**

> It is unlikely that every filter (or neuron) in a deep neural network needs to depend on the output of all the filters in the previous layer. In fact, reducing filter co-dependence in deep networks has been shown to benefit generalization.

> Instead of using a modified loss, regularization penalty, or randomized network connectivity during training to prevent co-adaption of features, we take a much more direct approach. We use filter groups (see Fig. 1) to force the network to learn filters with only limited dependence on previous layers. Each of the filters in the filter groups is smaller in the channel extent, since it operates on only a subset of the channels of the input feature map.

> This reduced connectivity also reduces computational complexity and model size since the size of filters in filter groups are reduced drastically, as is evident in Fig. 4. Unlike methods for increasing the efficiency of deep networks by approximating pre-trained existing networks (see §2), our models are trained from random initialization using stochastic gradient descent. This means that our method can also speed up training and, since we are not merely approximating an existing model's weights, the accuracy of the existing model is not an upper bound on accuracy of the modified model.

**Root Module**

Notations:
* Let $*$ denote the convolution operation.
* Let $c_{1} \in \mathbb{Z}_{++}$ denote the number of input channels to the layer.
* Let $c_{2} \in \mathbb{Z}_{++}$ denote the number of intermediate channels of the layer.
* Let $c_{3} \in \mathbb{Z}_{++}$ denote the number of output channels of the layer.
* Let $g \in \mathbb{Z}_{++}$ denote the number of filter groups.
The more filter groups, the fewer the number of connections to the previous layer's outputs.
* Let $h, w \in \mathbb{Z}_{++}$ denote the height and width of the CNN kernels.
* Let $K^{(ij)} \in \mathbb{R}^{h \times w \times c_{1}/g}$ denote the kernel that generates the $i$-th feature map in the $j$-th group, for $i \in \{1, ..., c_{2}/g\}$ and $j \in \{1, ..., g\}$.
* Let $L^{(1)}, ..., L^{(c_{3})} \in \mathbb{R}^{1 \times 1 \times c_{2}}$ denote the point-wise convolutional kernels.
* Let $H, W \in \mathbb{Z}_{++}$ denote the height and width of the input $x$.
* Let $x_{j} \in \mathbb{R}^{H \times W \times c_{1}/g}$ denote the $j$-th group of $x$, for $j \in \{1, ..., g\}$.

Then the root module $f: \mathbb{R}^{H \times W \times c_{1}} \to \mathbb{R}^{H \times W \times c_{3}}$ is given by
$$f(x) := \operatorname{Concat}\bigg\{\operatorname{Concat}\bigg\{x_{j} * K^{(ij)}: \ \begin{aligned}
& i \in \{1, ..., c_{2}/g\} \\
& j \in \{1, ..., g\}
\end{aligned}\bigg\} * L^{(k)}: k \in \{1, ..., c_{3}\}\bigg\}.$$

> Each spatial convolutional layer is followed by a low-dimensional embedding (1x1 convolution). Like in [9], this configuration learns a linear combination of the basis filters (filter groups), implicitly representing a filter of full channel depth, but with limited filter dependence.

## 4. Results

| Model              | Parameters | FLOPS     | CPU timing | GPU timing |
| ------------------ | ---------- | --------- | ---------- | ---------- |
| Network In Network | 33%        | 46%       | 37% faster | 23% faster |
| ResNet 50          | 27% fewer  | 37% fewer | 23% faster | 13% faster |
| GoogLeNet          | 7% fewer   | 44% fewer | 21% faster | 16% faster |

## 5. GPU Implementation

## 6. Future Work

> In this paper we focused on using homogeneous filter groups (with a uniform division of filters in each group), however this may not be optimal. Heterogeneous filter groups may reflect better the filter co-dependencies found in deep networks. Learning a combined spatial [9] and channel basis, may also improve efficiency further.

## 7. Conclusion

1. > We explored the effect of using complex hierarchical arrangements of filter groups in CNNs and show that imposing a structured decrease in the degree of filter grouping with depth - a 'root' (inverse tree) topology – can allow us to obtain more efficient variants of state-of-the-art networks without compromising accuracy.

2. > Our method appears to be complementary to existing methods, such as low-dimensional embeddings, and can be used more efficiently to train deep networks than methods that only approximate a pre-trained model's weights.

3. > We validated our method by using it to create more efficient variants of state-of-the-art Network-in-network, GoogLeNet, and ResNet architectures, which were evaluated on the CIFAR10 and ILSVRC datasets. Our results show similar accuracy with the baseline architecture with fewer parameters and much less compute (as measured by CPU and GPU timings).

----------------------------------------------------------------------------------------------------

## References

* Ioannou, Yani, et al. "Deep roots: Improving cnn efficiency with hierarchical filter groups." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017.

## Further Reading

* [2] [InceptionNetV1](https://zhuanlan.zhihu.com/p/564141144)
* [4] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [9] Training CNNs with Low-Rank Filters for Efficient Image Classification
* [19] Network In Network (NIN)
