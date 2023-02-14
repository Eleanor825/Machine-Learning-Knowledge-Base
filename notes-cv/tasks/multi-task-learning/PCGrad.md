# [Notes] PCGrad <!-- omit in toc -->

* Title: Gradient Surgery for Multi-Task Learning
* url: https://arxiv.org/abs/2001.06782

## Table of Contents <!-- omit in toc -->

- [1. Introduction](#1-introduction)
- [2. Multi-Task Learning with PCGrad](#2-multi-task-learning-with-pcgrad)
  - [2.2. The Tragic Triad: Conflicting Gradients, Dominating Gradients, High Curvature](#22-the-tragic-triad-conflicting-gradients-dominating-gradients-high-curvature)
  - [2.3. PCGrad: Project Conflicting Gradients](#23-pcgrad-project-conflicting-gradients)
  - [2.4. Theoretical Analysis of PCGrad](#24-theoretical-analysis-of-pcgrad)
- [See Also](#see-also)

----------------------------------------------------------------------------------------------------

## 1. Introduction

> In this work, we instead hypothesize that one of the main optimization issues in multi-task learning arises from gradients from different tasks conflicting with one another in a way that is detrimental to making progress.

> We define two gradients to be conflicting if they point away from one another, i.e., have a negative cosine similarity.

> We hypothesize that such conflict is detrimental when a) conflicting gradients coincide with b) high positive curvature and c) a large difference in gradient magnitudes.

> The core contribution of this work is a method for mitigating gradient interference by altering the gradients directly, i.e. by performing “gradient surgery.” If two gradients are conflicting, we alter the gradients by projecting each onto the normal plane of the other, preventing the interfering components of the gradient from being applied to the network.

## 2. Multi-Task Learning with PCGrad

### 2.2. The Tragic Triad: Conflicting Gradients, Dominating Gradients, High Curvature

> **Definition (Conflicting Gradients).**<br>
> Let $\phi_{ij}$ denote the angle between two task gradients $g_{i}$ and $g_{j}$.
> We say that $g_{i}$ and $g_{j}$ are conflicting if and only if $\cos\phi_{ij} < 0$.

> **Definition (Gradient Magnitude Similarity).**<br>
> We define the **gradient magnitude similarity** between two gradients $g_{i}$ and $g_{j}$, denoted by $\Phi(g_{i}, g_{j})$, to be
> $$\Phi(g_{i}, g_{j}) := \frac{2\|g_{i}\|_{2}\|g_{j}\|_{2}}{\|g_{i}\|_{2}^{2}+\|g_{j}\|_{2}^{2}}.$$

> **Proposition.**<br>
> We have the following:
> * $\Phi(g_{i}, g_{j}) \in [0, 1]$.
> * $\Phi(g_{i}, g_{j}) = \Phi(g_{j}, g_{i})$.
> * If any of $g_{i}$ or $g_{j}$ is zero, then $\Phi(g_{i}, g_{j}) = 0$.
> * If $\frac{\|g_{i}\|_{2}}{\|g_{j}\|_{2}} \to 0$, then $\Phi(g_{i}, g_{j}) \to 0$.
> * If $\frac{\|g_{i}\|_{2}}{\|g_{j}\|_{2}} = 1$, then $\Phi(g_{i}, g_{j}) = 1$.

> **Definition (First/Second-Order Directional Gradient).**<br>
> Let $f: \mathbb{R}^{n} \to \mathbb{R}$ and $\bar{x}, d \in \mathbb{R}^{n}$.
> Define $\tilde{f}: [0, +\infty) \to \mathbb{R}$ by $\tilde{f}(\alpha) := f(\bar{x}+\alpha d)$.
> We define the **first-order gradient** of $f$ along direction $d$ at point $\bar{x}$ is
> $$\nabla{f}(\bar{x}; d) := \tilde{f}'(0) = d^{\top}\nabla{f}(\bar{x}),$$
> and the **second-order gradient** of $f$ along direction $d$ at point $\bar{x}$ is
> $$\nabla^{2}f(\bar{x}; d) := \tilde{f}''(0) = d^{\top}[\nabla^{2}f(\bar{x})]d.$$

> **Definition (Multi-Task Curvature).**<br>
> Let $\theta_{1}, \theta_{2}$ be two sets of parameters.
> We define the **multi-task curvature** $H(\mathcal{L}; \theta_{1}, \theta_{2})$ to be
> $$H(\mathcal{L}; \theta_{1}, \theta_{2}) := \int_{0}^{1}[\nabla{\mathcal{L}}(\theta_{1})]^{\top}[\nabla^{2}\mathcal{L}((1-\alpha)\theta_{1}+\alpha\theta_{2})][\nabla{\mathcal{L}}(\theta_{1})]d\alpha,$$
> which is the averaged curvature of $\mathcal{L}$ between $\theta_{1}$ and $\theta_{2}$ in the direction of the multi-task gradient $\nabla{\mathcal{L}}(\theta_{1})$.

### 2.3. PCGrad: Project Conflicting Gradients

### 2.4. Theoretical Analysis of PCGrad

----------------------------------------------------------------------------------------------------

## See Also
