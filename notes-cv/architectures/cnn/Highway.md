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
$$y := H(x, W_{H}) \odot T(x, W_{T}) + x \odot C(x, W_{C}).$$

## Experiments

### 3.1 Optimization

> Highway networks do not suffer from an increase in depth, and 50/100 layer highway networks perform similar to 10/20 layer networks. The 100-layer highway network performed more than 2 orders of magnitude better compared to a similarly-sized plain network.

> It was also observed that highway networks consistently converged significantly faster than plain ones.

## Further Reading

* Maxout Networks
* Deeply Supervised Nets (DSN)
