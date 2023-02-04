# [Notes][Vison][Diffusion] Denoising Diffusion Implicit Models <!-- omit in toc -->

* urls: [[abs](https://arxiv.org/abs/2010.02502)]
    [[pdf](https://arxiv.org/pdf/2010.02502.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2010.02502/)]
* Title: Denoising Diffusion Implicit Models
* Year: 06 Oct `2020`
* Authors: Jiaming Song, Chenlin Meng, Stefano Ermon
* Institution: [Stanford University]
* Abstract: Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a Markovian diffusion process. We construct a class of non-Markovian diffusion processes that lead to the same training objective, but whose reverse process can be much faster to sample from. We empirically demonstrate that DDIMs can produce high quality samples $10 \times$ to $50 \times$ faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, and can perform semantically meaningful image interpolation directly in the latent space.

## Table of Contents <!-- omit in toc -->

- [Summary of Main Contributions](#summary-of-main-contributions)
- [1. Introduction](#1-introduction)
- [2. Background](#2-background)
- [3. Variational Inference for Non-Markovian Forward Processes](#3-variational-inference-for-non-markovian-forward-processes)
  - [Variational Inference Objective](#variational-inference-objective)
- [4. Sampling from Generalized Generative Processes](#4-sampling-from-generalized-generative-processes)
- [5. Experiments](#5-experiments)
- [References](#references)

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Generalized Markovian process in DDPM to non-Markovian
* Faster than DDPM by reducing the number of steps during sampling

----------------------------------------------------------------------------------------------------

## 1. Introduction

> To close this efficiency gap between DDPMs and GANs, we present denoising diffusion implicit models (DDIMs). DDIMs are implicit probabilistic models (Mohamed & Lakshminarayanan, 2016) and are closely related to DDPMs, in the sense that they are trained with the same objective function.

## 2. Background

**DDPM - Forward Process**

Given hyperparameters $\beta_{1}, ..., \beta_{T} \in (0, 1)$, we add noise to the original image sequentially.
$$q(x_{t}|x_{t-1}) := \mathcal{N}(\sqrt\frac{\alpha_{t}}{\alpha_{t-1}}x_{t-1}, (1-\frac{\alpha_{t}}{\alpha_{t-1}})I).$$
Note that this process is Markovian.
Formula for computing for arbitrary $t$ directly.
$$q(x_{t}|x_{0}) = \mathcal{N}(\sqrt{\alpha_{t}}x_{0}, (1-\alpha_{t})I).$$
Reparameterization:
$$x_{t} = \sqrt{\alpha_{t}}x_{0} + \sqrt{1-\alpha_{t}}\epsilon \text{ where } \epsilon \sim \mathcal{N}(0, I).$$

Reverse the process using Bayes' Theorem:
\begin{aligned}
    & q(x_{t-1}|x_{t}) = \mathcal{N}(\tilde{\mu}_{t}, \tilde{\beta}_{t}I) \text{ where } \\
    & \tilde{\mu}_{t} = \tilde{\mu}_{t}(x_{t}, x_{0}) := \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}x_{0} + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} \text{ and } \\
    & \tilde{\beta}_{t} := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t}.
\end{aligned}
We aim to model this using
$$p_{\theta}(x_{t-1}|x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t)).$$

**Training objective**

**Limitation**

> The length $T$ of the forward process is an important hyperparameter in DDPMs. From a variational perspective, a large $T$ allows the reverse process to be close to a Gaussian (Sohl-Dickstein et al., 2015), so that the generative process modeled with Gaussian conditional distributions becomes a good approximation; this motivates the choice of large $T$ values, such as $T=1000$ in Ho et al. (2020). However, as all $T$ iterations have to be performed sequentially, instead of in parallel, to obtain a sample $x_{0}$, sampling from DDPMs is much slower than sampling from other deep generative models, which makes them impractical for tasks where compute is limited and latency is critical.

## 3. Variational Inference for Non-Markovian Forward Processes

Let us consider a family $\mathcal{Q}$ of inference distributions, indexed by a real vector $\sigma \in \mathbb{R}^{T}_{+}$.

We still have
$$q(x_{t}|x_{0}) = \mathcal{N}(\sqrt{\bar{\alpha}_{t}}x_{0}, (1-\bar{\alpha}_{t})I).$$

Something different: non-Markovian
$$q_{\sigma}(x_{t}|x_{t-1}, x_{0}) = \frac{q_{\sigma}(x_{t-1}|x_{t}, x_{0})q_{\sigma}(x_{t}|x_{0})}{q_{\sigma}(x_{t-1}|x_{0})}.$$

If we take $\sigma_{t} := \sqrt{(1-\alpha_{t-1})/(1-\alpha_{t})}\sqrt{1-\alpha_{t}/\alpha_{t-1}}$, then we reduce to the case of the original DDPM.

### Variational Inference Objective

$$J_{\sigma}(\epsilon_{\theta}) := \mathbb{E}_{x_{0:T}\sim q_{\sigma}(x_{0:T})}\bigg[\log q_{\sigma}(x_{1:T}|x_{0}) - \log p_{\theta}(x_{0:T})\bigg].$$

**Theorem 1**.
For all $\sigma > 0$, there exists $\gamma \in \mathbb{R}^{T}_{++}$ and $C \in \mathbb{R}$ such that $J_{\sigma} = L_{\gamma} + C$.

## 4. Sampling from Generalized Generative Processes

> However, as the denoising objective $L_{1}$ does not depend on the specific forward procedure as long as $q_{\sigma}(x_{t}|x_{0})$ is fixed, we may also consider forward processes with lengths smaller than $T$, which accelerates the corresponding generative processes without having to train a different model.

Sample a subsequence $\tau = [\tau_{1}, ..., \tau_{S}] \subseteq [1, ..., T]$.
Then
\begin{aligned}
    & q(x_{\tau_{i}}|x_{0}) = \mathcal{N}(x_{t}; \sqrt{\alpha_{\tau_{i}}}x_{0}, (1-\alpha_{\tau_{i}})I) \text{ and } \\
    & q_{\sigma, \tau}(x_{1:T}|x_{0}) = q_{\sigma, \tau}(x_{T}|x_{0})\prod_{i=1}^{S}q_{\sigma}(x_{\tau_{i-1}}|x_{\tau_{i}}, x_{0})\prod_{t\in\bar{\tau}}q_{\sigma, \tau}(x_{t}|x_{0}).
\end{aligned}

> The generative process now samples latent variables according to reversed($\tau$), which we term *(sampling) trajectory*. When the length of the sampling trajectory is much smaller than $T$, we may achieve significant increases in computational efficiency due to the iterative nature of the sampling process.

## 5. Experiments

Consider $\sigma$ defined by
$$\sigma_{\tau_{i}}(\eta) := \eta\sqrt{(1-\alpha_{\tau_{i-1}})/(1-\alpha_{\tau_{i}})}\sqrt{1-\alpha_{\tau_{i}}/\alpha_{\tau_{i-1}}}.$$

> This includes an original DDPM generative process when $\eta = 1$ and DDIM when $\eta = 0$.

$\dim(\tau)$ represents the number of timesteps used to generate a sample.

> As expected, the sample quality becomes higher as we increase $\dim(\tau)$, presenting a tradeoff between sample quality and computational costs.

$\eta$ represents the stochasticity of the process.

<!-- 
## 6. Related Work

## 7. Discussion
-->

----------------------------------------------------------------------------------------------------

## References

* Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." *arXiv preprint arXiv:2010.02502* (2020).
* https://mp.weixin.qq.com/s/k7mSNHWFVQTUQBWZixHMuQ

<!-- ## Further Reading -->

