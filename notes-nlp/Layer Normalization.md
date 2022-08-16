# Layer Normalization

* Year: 21 Jul `2016`
* Author: Jimmy Lei Ba
* Abstract: Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.

----------------------------------------------------------------------------------------------------

## 1 Introduction

## 2 Background

> One of the challenges of deep learning is that the gradients with respect to the weights in one layer are highly dependent on the outputs of the neurons in the previous layer especially if these outputs change in a highly correlated way. Batch normalization was proposed to reduce such undesirable "covariate shift". The method normalizes the  summed inputs to each hidden unit over the training cases. Specifically, for the $i^{\text{th}}$ summed input in the $l^{\text{th}}$ layer, the batch normalization method rescales the summed inputs according to their variances under the distribution of the data
> $$\bar{a}_{i}^{l} := \frac{g_{i}^{l}}{\sigma_{i}^{l}}(a_{i}^{l} - \mu_{i}^{l}) \quad\quad
\mu_{i}^{l} := \underset{\textbf{x} \sim P(\textbf{x})}{\mathbb{E}}[a_{i}^{l}] \quad\quad
\sigma_{i}^{l} := \sqrt{\underset{\textbf{x} \sim P(\textbf{x})}{\mathbb{E}}[(a_{i}^{l} - \mu_{i}^{l})^{2}]} \tag{2}$$
> where $\bar{a}_{i}^{l}$ is normalized summed inputs to the $i^{\text{th}}$ hidden unit in the $l^{\text{th}}$ layer and $g_{i}$ is a gain parameter scaling the normalized activation before the non-linear activation function. Note the expectation is under the whole training data distribution. It is typically impractical to compute the expectations in Eq. (2) exactly, since it would require forward passes through the whole training dataset with the current set of weights. Instead, $\mu$ and $\sigma$ are estimated using the empirical samples from the current mini-batch. This puts constraints on the size of a mini-batch and it is hard to apply to recurrent neural networks.

## 3 Layer normalization

> Notice that changes in the output of one layer will tend to cause highly correlated changed in the  summed inputs to the next layer, especially with ReLU units whose outputs can change by a log. This suggests the "covariate shift" problem can be reduced by fixing the mean and the variance of the summed inputs within each layer. We, thus, compute the layer  normalization statistics over all the hidden units in the same layer as follows:
> $$\mu^{l} := \frac{1}{H}\sum_{i=1}^{H}a{i}^{l} \quad\quad
\sigma^{l} := \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a_{i}^{l} - \mu_{l})^{2}} \tag{3}$$
where $H$ denotes the number of hidden units in a layer. The difference between Eq. (2) and Eq. (3) is that  under layer normalization, all the hidden units in a layer share the same normalization terms $\mu$ and $\sigma$, but different training cases have different normalization term. Unlike batch normalization, layer normalization does not impose any constraint on the size of a mini-batch and it can be used in the pure online regime with batch size 1.

### 3.1 Layer normalized recurrent neural networks

> It is common among the NLP tasks to have different sentence lengths for different training cases. This is easy to deal with in an RNN because the same weights are used at every time-step. But when we apply batch normalization to an RNN in the obvious way, we need to compute and store separate statistics for each time step in a sequence. This is problematic if a test sequence is longer than any of the training sequences. Layer normalization does not have such problem because its normalization terms depend only on the summed inputs to a layer at the current time-step. It also has only one set of gain and bias parameters shared over all time-steps.

> In a standardRNN, the summed inputs in the recurrent layer are computed from the current input $\textbf{x}^{t}$ and previous vector of hidden states $\textbf{h}^{t-1}$ which are computed as $\textbf{a}^{t} = W_{hh}h^{t-1}  + W_{xh}\textbf{x}^{t}$. The layer normalized recurrent layer re-centers and re-scales its activations using the extra normalization terms similar to Eq. (3):
> $$ \textbf{h}^{t} := f\bigg[\frac{\textbf{g}}{\sigma^{t}} \odot (\textbf{a}^{t} - \mu^{t}) + \textbf{b}\bigg] \quad\quad
\mu^{t} := \frac{1}{H}\sum_{i=1}^{H}a_{i}^{t} \quad\quad
\sigma^{t} := \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a_{i}^{t} - \mu^{t})^{2}} \tag{4}$$
where $W_{hh}$ is the recurrent hidden to hidden weights and $W_{xh}$ are the bottom up input to hidden weights.
$\textbf{b}$ and $\textbf{g}$ are defined as the bias and gain parameters of the same dimension as $\textbf{h}^{t}$.

> In a standard RNN, there is a tendency for the average magnitude of the summed inputs to the recurrent units to either grow or shrink at every time-step, leading to exploding or vanishing gradients. In a layer normalized RNN, the normalization terms make it invariant to re-scaling all of the summed inputs toa layer, which results in much more stable hidden-to-hidden dynamics.

## 4 Related work

