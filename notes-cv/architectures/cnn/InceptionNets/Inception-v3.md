# [Notes][Vision][CNN] Inception-v3

* url: https://arxiv.org/abs/1512.00567
* Title: Rethinking the Inception Architecture for Computer Vision
* Year: 02 Dec `2015`
* Authors: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
* Institutions: [Google Inc.], [University College London]
* Abstract: Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set.

----------------------------------------------------------------------------------------------------

## 1. Introduction

## 2. General Design Principles

> 1. Avoid representational bottlenecks, especially early in the network.
> One should avoid bottlenecks with extreme compression. In general the representation size should gently decrease from the inputs to the outputs before reaching the final representation used for the task at hand.

> 2. Higher dimensional representations are easier to process locally within a network.

> 3. Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.
> We hypothesize that the reason for that is the strong correlation between adjacent unit results in much less loss of information during dimension reduction, if the outputs are used in a spatial aggregation context. Given that these signals should be easily compressible, the dimension reduction even promotes faster learning.

> 4. Balance the width and depth of the network.
> Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network.

## 3. Factorizing Convolutions with Large Filter Size

### 3.1. Factorization into smaller convolutions

> Convolutions with larger spatial filters (e.g. 5x5 or 7x7) tend to be disproportionally expensive in terms of computation.

> Of course, a 5x5 filter can capture dependencies between signals between activations of units further away in the earlier layers, so a reduction of the geometric size of the filters comes at a large cost of expressiveness. However, we can ask whether a 5x5 convolution could be replaced by a multi-layer network with less parameters with the same input size and output depth.

Notations:
* Let $n \in \mathbb{Z}_{++}$ denote the number of input channels.
* Let $m \in \mathbb{Z}_{++}$ denote the number of output channels.
* Let $K_{1} \in \mathbb{R}^{n \times 5 \times 5 \times m}$ denote a 5x5 kernel.
* Let $K_{2} \in \mathbb{R}^{n \times 3 \times 3 \times \sqrt{mn}}$ denote the first 3x3 kernel.
* Let $K_{3} \in \mathbb{R}^{\sqrt{mn} \times 3 \times 3 \times m}$ denote the second 3x3 kernel.

The dimension reduction factor of a single 5x5 kernel is $\alpha := n / m$.

The dimension reduction factor of both of the 3x3 kernels is $\beta := n / \sqrt{mn} = \sqrt{\alpha}$.

> Still, this setup raises two general questions:
> 1. Does this replacement result in any loss of expressiveness?
> 2. If our main goal is to factorize the linear part of the computation, would it not suggest to keep linear activations in the first layer?

> We have ran several control experiments (for example see figure 2) and using linear activation was always inferior to using rectified linear units in all stages of the factorization. We attribute this gain to the enhanced space of variations that the network can learn especially if we batch-normalize [7] the output activations.

### 3.2. Spatial Factorization into Asymmetric Convolutions

> using a 3x1 convolution followed by a 1x3 convolution is equivalent to sliding a two layer network with the same receptive field as in a 3x3 convolution (see figure 3).

> In practice, we have found that employing this factorization does not work well on early layers, but it gives very good results on medium grid-sizes (On $m \times m$ feature maps, where $m$ ranges between 12 and 20).

## 4. Utility of Auxiliary Classifiers

> Interestingly, we found that auxiliary classifiers did not result in improved convergence early in the training: the training progression of network with and without side head looks virtually identical before both models reach high accuracy. Near the end of training, the network with the auxiliary branches starts to overtake the accuracy of the network without any auxiliary branch and reaches a slightly higher plateau.

> The removal of the lower auxiliary branch did not have any adverse effect on the final quality of the network. Together with the earlier observation in the previous paragraph, this means that original the hypothesis of [20] that these branches help evolving the low-level features is most likely misplaced.

> Instead, we argue that the auxiliary classifiers act as regularizer. This is supported by the fact that the main classifier of the network performs better if the side branch is batch-normalized [7] or has a dropout layer.

## 5. Efficient Grid Size Reduction



----------------------------------------------------------------------------------------------------

## References

* Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

## Further Reading

* [5] R-CNN
* [7] Inception-v2/Batch Normalization
* [9] AlexNet
* [12] Fully Convolutional Networks (FCN)
* [18] VGG
* [20] Inception-v1/GoogLeNet
