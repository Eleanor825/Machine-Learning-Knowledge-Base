#! https://zhuanlan.zhihu.com/p/580591950
# [Notes][Vision][GAN] Generative Adversarial Networks <!-- omit in toc -->

* urls: [[abs](https://arxiv.org/abs/1406.2661)]
    [[pdf](https://arxiv.org/pdf/1406.2661.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1406.2661/)]
* Title: Generative Adversarial Networks
* Year: 10 Jun `2014`
* Authors: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
* Abstract: We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

## Table of Contents <!-- omit in toc -->

- [Summary of Main Contributions](#summary-of-main-contributions)
- [1 Introduction](#1-introduction)
- [2 Related work](#2-related-work)
- [3 Adversarial nets](#3-adversarial-nets)
- [4 Theoretical Results](#4-theoretical-results)
- [5 Experiments](#5-experiments)
- [6 Advantages and disadvantages](#6-advantages-and-disadvantages)
- [7 Conclusions and future work](#7-conclusions-and-future-work)
- [References](#references)
- [Further Reading](#further-reading)

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

----------------------------------------------------------------------------------------------------

## 1 Introduction

> So far, the most striking successes in deep learning have involved `discriminative` models, usually those that map a high-dimensional, rich sensory input to a class label [14, 22].

> In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution. The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistinguishable from the genuine articles.

## 2 Related work

## 3 Adversarial nets

> In practice, equation 1 may not provide sufficient gradient for $G$ to learn well. Early in learning, when $G$ is poor, $D$ can reject samples with high confidence because they are clearly different from the training data. In this case, $\log(1-D(G(z)))$ saturates. Rather than training $G$ to minimize $\log(1-D(G(z)))$ we can train $G$ to maximize $\log(D(G(z)))$. This objective function results in the same fixed point of the dynamics of $G$ and $D$ but provides much stronger gradients early in learning.

## 4 Theoretical Results

## 5 Experiments

## 6 Advantages and disadvantages

> The disadvantages are primarily that there is no explicit representation of $p_{g}(x)$, and that $D$ must be synchronized well with $G$ during training (in particular, $G$ must not be trained too much without updating $D$, in order to avoid "the Helvetica scenario" in which $G$ collapses too many values of $z$ to the same value of $x$ to have enough diversity to model $p_{data}$), much as the negative chains of a Boltzmann machine must be kept up to date between learning steps.

> 1. The advantages are that Markov chains are never needed, only backprop is used to obtain gradients, no inference is needed during learning, and a wide variety of functions can be incorporated into the model.
> 2. Adversarial models may also gain some statistical advantage from the generator network not being updated directly with data examples, but only with gradients flowing through the discriminator. This means that components of the input are not copied directly into the generator's parameters.
> 3. Another advantage of adversarial networks is that they can represent very sharp, even degenerate distributions, while methods based on Markov chains require that the distribution be somewhat blurry in order for the chains to be able to mix between modes.

## 7 Conclusions and future work

----------------------------------------------------------------------------------------------------

## References

* Goodfellow, Ian, et al. "Generative adversarial networks." *Communications of the ACM* 63.11 (2020): 139-144.

----------------------------------------------------------------------------------------------------

## Further Reading

* [22] AlexNet
* [10] Maxout
* [17] Dropout
