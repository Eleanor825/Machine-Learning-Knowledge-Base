# [Notes][Vision][CNN] Inception-v2/Batch Normalization

* url: https://arxiv.org/abs/1502.03167
* Title: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
* Year: 11 Feb `2015`
* Authors: Sergey Ioffe, Christian Szegedy
* Institutions: [Google Inc.]
* Abstract: Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

----------------------------------------------------------------------------------------------------

## 1 Introduction

> Using mini-batches of examples, as opposed to one example at a time, is helpful in several ways. First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than m computations for individual examples, due to the parallelism afforded by the modern computing platforms.

> The training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers – so that small changes to the network parameters amplify as the network becomes deeper.

> The change in the distributions of layers' inputs presents a problem because the layers need to continuously adapt to the new distribution. When the input distribution to a learning system changes, it is said to experience covariate shift (Shimodaira, 2000).

> Fixed distribution of inputs to a sub-network would have positive consequences for the layers outside the subnetwork, as well.

> We refer to the change in the distributions of internal nodes of a deep network, in the course of training, as Internal Covariate Shift. Eliminating it offers a promise of faster training.

**Benefits of Batch Normalization**

> We propose a new mechanism, which we call Batch Normalization, that takes a step towards reducing internal covariate shift, and in doing so
> 1. dramatically accelerates the training of deep neural nets. It accomplishes this via a normalization step that fixes the means and variances of layer inputs.
> 2. Batch Normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows us to use much higher learning rates without the risk of divergence.
> 3. Furthermore, batch normalization regularizes the model and reduces the need for Dropout (Srivastava et al., 2014).
> 4. Finally, Batch Normalization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes.

## 2 Towards Reducing Internal Covariate Shift

> It has been long known (LeCun et al., 1998b; Wiesler & Ney, 2011) that the network training converges faster if its inputs are whitened – i.e., linearly transformed to have zero means and unit variances, and decorrelated.

Notations:
* Let $x \in \mathbb{R}^{n}$ denote the input to a layer.
* Let $\mathcal{X}$ denote the set of inputs to the layer, yield from the training dataset.

> Since the full whitening of each layer's inputs is costly and not everywhere differentiable, we make two necessary simplifications.

**Simplification 1**

> The first is that instead of whitening the features in layer inputs and outputs jointly, we will normalize each scalar feature independently, by making it have the mean of zero and the variance of 1.

Notations:
* Let $x \in \mathbb{R}^{d}$ denote the input.
* Let $\hat{x} \in \mathbb{R}^{d}$ denote the normalized vector.

Then $\forall k \in \{1, ..., d\}$,
$$\hat{x}_{k} := \frac{x_{k} - \mathbb{E}[x_{k}]}{\sqrt{\mathbb{V}[x_{k}]}}$$
where the expectation and variance are computed over the training data set.

> Note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. To address this, we make sure that the transformation inserted in the network can represent the identity transform.

> To accomplish this, we introduce, for each activation $x_{k}$, a pair of parameters $\gamma_{k}$, $\beta_{k}$, which scale and shift the normalized value:
$$y_{k} := \gamma_{k}\hat{x}_{k} + \beta_{k}.$$

> These parameters are learned along with the original model parameters, and restore the representation power of the network. Indeed, by setting $\gamma_{k} := \sqrt{\mathbb{V}[x_{k}]}$ and $\beta_{k} := \mathbb{E}[x_{k}]$, we could recover the original activations,
if that were the optimal thing to do.

**Simplification 2**

> We make the second simplification: since we use mini-batches in stochastic gradient training, each mini-batch produces estimates
of the mean and variance of each activation. This way, the statistics used for normalization can fully participate in the gradient backpropagation.

Notations:
* Let $d \in \mathbb{Z}_{++}$ denote the dimension of the data points.
* Let $m \in \mathbb{Z}_{++}$ denote the size of the mini-batch.
* Let $\mathcal{B} = \{x_{1}, ..., x_{m}\} \subseteq \mathbb{R}^{d}$ denote a mini-batch.
* Let $\hat{x}^{(1)}, ..., \hat{x}^{(m)} \in \mathbb{R}^{d}$ denote the normalized values.
* Let $y^{(1)}, ..., y^{(m)} \in \mathbb{R}^{d}$ denote the linear transformations of $x^{(1)}, ..., x^{(m)}$.

Then the batch normalizaton layer $\operatorname{BN}_{\gamma, \beta}: \bigoplus_{i=1}^{m}\mathbb{R}^{d} \to \bigoplus_{i=1}^{m}\mathbb{R}^{d}$, parameterized by $\gamma \in \mathbb{R}^{d}$ and $\beta \in \mathbb{R}^{d}$, is given by:
$$\begin{align*}
    \text{ (mini-batch mean) } \quad &
        \mu_{\mathcal{B}} := \frac{1}{m}\sum_{i=1}^{m}x^{(i)} \in \mathbb{R}^{d} \\
    \text{ (mini-batch variance) } \quad &
        \sigma^{2}_{\mathcal{B}} := \frac{1}{m}\sum_{i=1}^{m}(x^{(i)} - \mu_{\mathcal{B}})^{2} \in \mathbb{R}^{d} \\
    \text{ (normalize) } \quad &
        \hat{x}^{(i)} := \frac{x^{(i)} - \mu_{\mathcal{B}}}{\sqrt{\sigma^{2}_{\mathcal{B}} + \varepsilon}} \in \mathbb{R}^{d}, \quad
        \forall i \in \{1, ..., m\} \\
    \text{ (transform) } \quad &
        y^{(i)} := \gamma\hat{x}^{(i)} + \beta \in \mathbb{R}^{d}, \quad
        \forall i \in \{1, ..., m\} \\
    \text{ (output) } \quad &
        \operatorname{BN}_{\gamma, \beta}(x^{(1)}, ..., x^{(m)}) := (y^{(i)})_{i=1}^{m} \in \bigoplus_{i=1}^{m}\mathbb{R}^{d}. \\
    \end{align*}$$

----------------------------------------------------------------------------------------------------

## References

* Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *International conference on machine learning*. PMLR, 2015.

## Further Reading

