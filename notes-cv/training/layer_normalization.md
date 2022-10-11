# [Notes][Vision][Training] Layer Normalization

* url: https://arxiv.org/abs/1607.06450
* Title: Layer Normalization
* Year: 21 Jul `2016`
* Authors: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
* Institutions: [University of Toronto], [Google Inc.]
* Abstract: Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

----------------------------------------------------------------------------------------------------

## 1 Introduction

<!-- $$\begin{align*}
    \mu & = \frac{1}{H}\sum_{i=1}^{H}W_{i, \cdot}x \\
    \mu' & = \frac{1}{H}\sum_{i=1}^{H}(\delta W_{i, \cdot} + \gamma^{\top})x \\
         & = \delta\frac{1}{H}\sum_{i=1}^{H}W_{i, \cdot}x + \frac{1}{H}\sum_{i=1}^{H}\gamma^{\top}x \\
         & = \delta\mu + \gamma^{\top}x
    \end{align*}$$ -->

----------------------------------------------------------------------------------------------------

## References

* Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." *arXiv preprint arXiv:1607.06450* (2016).

## Further Reading

* [Krizhevsky et al., 2012] AlexNet
* [Ioffe and Szegedy, 2015] InceptionNetV2/Batch Normalization
