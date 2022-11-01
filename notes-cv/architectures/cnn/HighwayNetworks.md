#! https://zhuanlan.zhihu.com/p/554615809
# [Notes][Vision][CNN] Highway Networks

* url: https://arxiv.org/abs/1507.06228
* Title: Training Very Deep Networks
* Year: 22 Jul `2015`
* Authors: Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
* Abstract: Theoretical and empirical evidence indicates that the depth of neural networks is crucial for their success. However, training becomes more difficult as depth increases, and training of very deep networks remains an open problem. Here we introduce a new architecture designed to overcome this. Our so-called highway networks allow unimpeded information flow across many layers on information highways. They are inspired by Long Short-Term Memory recurrent networks and use adaptive gating units to regulate the information flow. Even with hundreds of layers, highway networks can be trained directly through simple gradient descent. This enables the study of extremely deep and efficient architectures.

----------------------------------------------------------------------------------------------------

## 1 Introduction & Previous Work

> To overcome this, we take inspiration from Long Short Term Memory (LSTM) recurrent networks [29, 30]. We propose to modify the architecture of very deep feedforward networks such that information flow across layers becomes much easier. This is accomplished through an LSTM-inspired adaptive gating mechanism that allows for computation paths along which information can flow across many layers without attenuation. We call such paths information highways. They yield highway networks, as opposed to traditional 'plain' networks.

## 2 Highway Networks

Notations:
* Let $\odot$ denote the element-wise product.
* Let $x \in \mathbb{R}^{d}$ denote the input to a layer.
* Let $y \in \mathbb{R}^{d}$ denote the output of a layer.
* Let $H: \mathbb{R}^{d} \oplus \mathbb{R}^{d \times d} \to \mathbb{R}^{d}$ denote the non-linear transformation of the original layer.
* Let $T: \mathbb{R}^{d} \oplus \mathbb{R}^{d \times d} \to \mathbb{R}^{d}$ denote the *transform* gate.
* Let $C: \mathbb{R}^{d} \oplus \mathbb{R}^{d \times d} \to \mathbb{R}^{d}$ denote the *carry* gate.

Then the Highway block $\mathbb{R}^{d} \to \mathbb{R}^{d}$ is defined by:
$$y := H(x, W_{H}) \odot T(x, W_{T}) + x \odot C(x, W_{C}). \tag{3}$$

## 2.1 Constructing Highway Networks

> As mentioned earlier, Equation 3 requires that the dimensionality of $x$, $y$, $H(x, W_{H})$, and $T(x, W_{T})$ be the same. To change the size of the intermediate representation, one can replace $x$ with $\hat{x}$ obtained by suitably sub-sampling or zero-padding $x$. Another alternative is to use a plain layer (without highways) to change dimensionality, which is the strategy we use in this study.

## 2.2 Training Deep Highway Networks

> In our experiments, we found that a negative bias initialization for the transform gates was sufficient for training to proceed in very deep networks for various zero-mean initial distributions of $W_{H}$ and different activation functions used by $H$.

## 3 Experiments

### 3.1 Optimization

> Highway networks do not suffer from an increase in depth, and 50/100 layer highway networks perform similar to 10/20 layer networks. The 100-layer highway network performed more than 2 orders of magnitude better compared to a similarly-sized plain network.

> It was also observed that highway networks consistently converged significantly faster than plain ones.

## 4 Analysis

### 4.1 Routing of Information

> One possible advantage of the highway architecture over hard-wired shortcut connections is that the network can learn to dynamically adjust the routing of the information based on the current input.

## 4.2 Layer Importance

> For each layer, we evaluated the network on the full training set with the gates of that layer closed.

> For MNIST (left) it can be seen that the error rises significantly if any one of the early layers is removed, but layers 15-45 seem to have close to no effect on the final performance. About 60% of the layers don't learn to contribute to the final result, likely because MNIST is a simple dataset that doesn't require much depth.

> We see a different picture for the CIFAR-100 dataset (right) with performance degrading noticeably when removing any of the first $\approx$ 40 layers. This suggests that for complex problems a highway
network can learn to utilize all of its layers, while for simpler problems like MNIST it will keep many of the unneeded layers idle. Such behavior is desirable for deep networks in general, but appears difficult to obtain using plain networks.

## 5 Discussion

> Very deep highway networks, on the other hand, can directly be trained with simple gradient descent methods due to their specific architecture. This property does not rely on specific non-linear transformations, which may be complex convolutional or recurrent transforms, and derivation of a suitable initialization scheme is not essential. The additional parameters required by the gating mechanism help in routing information through the use of multiplicative connections, responding differently to different inputs, unlike fixed "skip" connections.

----------------------------------------------------------------------------------------------------

## References

* Srivastava, Rupesh K., Klaus Greff, and Jürgen Schmidhuber. "Training very deep networks." *Advances in neural information processing systems* 28 (2015).

## Further Reading

* [20] Maxout Networks
* [24] Deeply-Supervised Nets (DSN)
* [25] FitNets
* [35] Network in Network (NIN)
