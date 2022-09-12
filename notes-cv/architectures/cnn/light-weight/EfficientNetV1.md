#! https://zhuanlan.zhihu.com/p/557565000
# [Notes][Vision][CNN] EfficientNetV1

* url: https://arxiv.org/abs/1905.11946
* Title: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
* Year: 28 May `2019`
* Authors: Mingxing Tan, Quoc V. Le
* Abstract: Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet. To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at [this https URL](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

----------------------------------------------------------------------------------------------------

## 1. Introduction

> `Scaling` up ConvNets is widely used to achieve better `accuracy`.

> The most common way is to scale up ConvNets by their `depth` (He et al., 2016) or `width` (Zagoruyko & Komodakis, 2016). Another less common, but increasingly popular, method is to scale up models by image `resolution` (Huang et al., 2018). In previous work, it is common to scale only one of the three dimensions – depth, width, and image size.

> Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency.

> In this paper, we want to study and rethink the process of scaling up ConvNets. In particular, we investigate the central question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency?

> Our empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio.

> Intuitively, the compound scaling method makes sense because if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

> Notably, the effectiveness of model scaling heavily depends on the `baseline network`; to go even further, we use neural architecture search (Zoph & Le, 2017; Tan et al., 2019) to develop a new baseline network, and scale it up to obtain a family of models, called `EfficientNets`.

## 2. Related Work

**ConvNet Accuracy**

**ConvNet Efficiency**

**Model Scaling**

> There are many ways to scale a ConvNet for different resource constraints:
> 1. ResNet (He et al., 2016) can be scaled down (e.g., ResNet-18) or up (e.g., ResNet-200) by adjusting `network depth` (#layers),
> 2. while WideResNet (Zagoruyko & Komodakis, 2016) and MobileNets (Howard et al., 2017) can be scaled by `network width` (#channels).
> 3. It is also well-recognized that bigger input image size will help accuracy with the overhead of more FLOPS.

> Our work systematically and empirically studies ConvNet scaling for all three dimensions of network `width`, `depth`, and `resolutions`.

## 3. Compound Model Scaling

### 3.1. Problem Formulation

Notations:
* Let $s \in \mathbb{Z}_{++}$ denote the number of stages.
* Let $H_{i}, W_{i} \in \mathbb{Z}_{++}$ denote the height and width of the input tensor to stage $i$ of the baseline network, for $i \in \{1, ..., s\}$.
* Let $C_{i} \in \mathbb{Z}_{++}$ denote the number of channels of the input tensor to stage $i$ of the baseline network, for $i \in \{1, ..., s\}$.
* Let $X_{\langle H_{i}, W_{i}, C_{i} \rangle} \in \mathbb{R}^{H_{i} \times W_{i} \times C_{i}}$ denote the input tensor to stage $i$ of the baseline network, for $i \in \{1, ..., s\}$.
* Let $\mathcal{F}_{i}: \mathbb{R}^{H_{i} \times W_{i} \times C_{i}} \to ?$ denote the convolutional layer to be repeated in stage $i$ of the baseline network, for $i \in \{1, ..., s\}$.
* Let $L_{i} \in \mathbb{Z}_{++}$ denote the number of times layer $\mathcal{F}_{i}$ is repeated in stage $i$ of the baseline network, for $i \in \{1, ..., s\}$.
* Let $\mathcal{F}_{i}^{L_{i}} := \underbrace{\mathcal{F}_{i} \circ ... \circ \mathcal{F}_{i}}_{L_{i}}$ denote stage $i$ of the baseline network.
* Let $\mathcal{N} := \bigodot_{i \in \{1, ..., s\}}\mathcal{F}_{i}^{L_{i}}(X_{\langle H_{i}, W_{i}, C_{i} \rangle})$ denote the baseline network.
* Define
$$\mathcal{N}(d, w, r) := \bigodot_{i \in \{1, ..., s\}}\mathcal{F}_{i}^{d \cdot L_{i}}(X_{\langle r \cdot H_{i}, r \cdot W_{i}, w \cdot C_{i} \rangle}).$$

Then our target is the following optimization problem:
$$\begin{aligned}
    \max_{d, w, r \in \mathbb{Z}_{++}} & \operatorname{Accuracy}(\mathcal{N}(d, w, r)) \\
    \text{subject to:} & \quad
    \begin{aligned}
        & \operatorname{Memory}(\mathcal{N}(d, w, r)) \leq \text{target\_memory} \\
        & \operatorname{FLOPS}(\mathcal{N}(d, w, r)) \leq \text{target\_flops}
    \end{aligned}
\end{aligned}$$

### 3.2. Scaling Dimensions

> Conventional methods mostly scale ConvNets in one of these dimensions:

**Depth ($d$)**

> Scaling network `depth` is the most common way used by many ConvNets (He et al., 2016; Huang et al., 2017; Szegedy et al., 2015; 2016). The intuition is that deeper ConvNet can capture richer and more complex features, and generalize well on new tasks.

> However, deeper networks are also more difficult to train due to the vanishing gradient problem (Zagoruyko & Komodakis, 2016). Although several techniques, such as skip connections (He et al., 2016) and batch normalization (Ioffe & Szegedy, 2015), alleviate the training problem, the accuracy gain of very deep network diminishes: for example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more layers.

**Width ($w$)**

> Scaling network `width` is commonly used for small size models (Howard et al., 2017; Sandler et al., 2018; Tan et al., 2019). As discussed in (Zagoruyko & Komodakis, 2016), wider networks tend to be able to capture more fine-grained features and are easier to train.

> However, extremely wide but shallow networks tend to have difficulties in capturing higher level features.

**Resolution ($r$)**

> With higher `resolution` input images, ConvNets can potentially capture more fine-grained patterns. Starting from 224x224 in early ConvNets, modern ConvNets tend to use 299x299 (Szegedy et al., 2016) or 331x331 (Zoph et al., 2018) for better accuracy. Recently, GPipe (Huang et al., 2018) achieves state-of-the-art ImageNet accuracy with 480x480 resolution. Higher resolutions, such as 600x600, are also widely used in object detection ConvNets (He et al., 2017; Lin et al., 2017).

> Indeed higher resolutions improve accuracy, but the accuracy gain diminishes for very high resolutions.

**Observation 1**

> Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.

### 3.3. Compound Scaling

> Intuitively, for higher resolution images, we should increase network depth, such that the larger receptive fields can help capture similar features that include more pixels in bigger images.

> Correspondingly, we should also increase network width when resolution is higher, in order to capture more fine-grained patterns with more pixels in high resolution images.

> These intuitions suggest that we need to coordinate and balance different scaling dimensions rather than conventional single-dimension scaling.

**Observation 2**

> In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.

> In this paper, we propose a new compound scaling method, which use a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a principled way:

$$\begin{aligned}
    \text{depth}: &&& d := \alpha^{\phi} \\
    \text{width}: &&& w := \beta^{\phi} \\
    \text{resolution}: &&& r := \gamma^{\phi} \\
    \text{subject to}: &&& \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2 \\
    &&& \alpha \geq 1, \beta \geq 1, \gamma \geq 1
\end{aligned}$$

> where $\alpha$, $\beta$, $\gamma$ are constants that can be determined by a small grid search.

> Intuitively, $\phi$ is a user-specified coefficient that controls how many more resources are available for model scaling, while $\alpha$, $\beta$, $\gamma$ specify how to assign these extra resources to network width, depth, and resolution respectively.

> Notably, the FLOPS of a regular convolution op is proportional to $d$, $w^{2}$, $r^{2}$.

> Since convolution ops usually dominate the computation cost in ConvNets, scaling a ConvNet with equation 3 will approximately increase total FLOPS by $(\alpha \cdot \beta^{2} \cdot \gamma^{2})^{\phi}$. In this paper, we constraint $\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$ such that for any new $\phi$, the total FLOPS will approximately increase by $2^{\phi}$.

## 4. EfficientNet Architecture

> Starting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps:
> * STEP 1: we first fix $\phi$ = 1, assuming twice more resources available, and do a small grid search of $\alpha$, $\beta$, $\gamma$ based on Equation 2 and 3. In particular, we find the best values for EfficientNet-B0 are $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$, under constraint of $\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$.
> * STEP 2: we then fix $\alpha$, $\beta$, $\gamma$ as constants and scale up baseline network with different $\phi$ using Equation 3, to obtain EfficientNet-B1 to B7 (Details in Table 2).

> Our method solves this issue by only doing search once on the small baseline network (step 1), and then use the same scaling coefficients for all other models (step 2).

## 5. Experiments

## 6. Discussion

## 7. Conclusion

----------------------------------------------------------------------------------------------------

## References

* Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." *International conference on machine learning*. PMLR, 2019.

## Further Reading

* [Krizhevsky et al., 2012] AlexNet
* [He et al., 2016] ResNet
* [Zagoruyko & Komodakis, 2016] Wide Residual Networks
* [Huang et al., 2017] DenseNet
* [Szegedy et al., 2015] Inception-v1/GoogLeNet
* [Ioffe & Szegedy, 2015] Inception-v2/Batch Normalization
* [Szegedy et al., 2016] Inception-v3
* [Szegedy et al., 2017] Inception-v4
* [Huang et al., 2018] GPipe
* [Howard et al., 2017] MobileNetV1
* [Sandler et al., 2018] MobileNetV2
* [Hu et al., 2018] SENet
* [Tan et al., 2019] MnasNet

* [EfficientNetV2](https://zhuanlan.zhihu.com/p/558323195)
