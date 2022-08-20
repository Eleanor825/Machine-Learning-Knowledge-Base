#! https://zhuanlan.zhihu.com/p/555110412
# [Notes][Vision][CNN] Maxout Networks
还没读透。。有机会再完善吧
* url: https://arxiv.org/abs/1302.4389
* Title: Maxout Networks
* Year: 18 Feb `2013`
* Authors: Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, Yoshua Bengio
* Abstract: We consider the problem of designing models to leverage a recently introduced approximate model averaging technique called dropout. We define a simple new model called maxout (so named because its output is the max of a set of inputs, and because it is a natural companion to dropout) designed to both facilitate optimization by dropout and improve the accuracy of dropout's fast approximate model averaging technique. We empirically verify that the model successfully accomplishes both of these tasks. We use maxout and dropout to demonstrate state of the art classification performance on four benchmark datasets: MNIST, CIFAR-10, CIFAR-100, and SVHN.

----------------------------------------------------------------------------------------------------

## 1. Introduction

## 2. Review of dropout

## 3. Description of maxout

> The maxout model is simply a feed-forward architecture, such as a multilayer perceptron or deep convolutional neural network, that uses a new type of activation function: the maxout unit.

Notations:
* Let $d \in \mathbb{Z}_{++}$ denote the dimension of the input to the layer.
* Let $m \in \mathbb{Z}_{++}$ denote the dimension of the output of the layer.
* Let $k \in \mathbb{Z}_{++}$ denote the number of affine approximators to the layer.
* Let $W^{(1)}, ..., W^{(k)} \in \mathbb{R}^{m \times d}$ denote the (learned) weights of the affine approximators.
* Let $b^{(1)}, ..., b^{(k)} \in \mathbb{R}^{m}$ denote the (learned) biases of the affine approximators.

Then the maxout hidden layer $h: \mathbb{R}^{d} \to \mathbb{R}^{m}$ is given by
$$h(x) := \max_{j \in [1, k]}(W^{(j)}x + b^{(j)})$$
where the max is taken elementwise.

> When training with dropout, we perform the elementwise multiplication with the dropout mask immediately prior to the multiplication by the weights in all cases - we do not drop inputs to the max operator.
> A single maxout unit can be interpreted as making a piecewise linear approximation to an arbitrary convex function.
> Maxout networks learn not just the relationship between hidden units, but also the activation function of each hidden unit.

## 4. Maxout is a universal approximator

**Proposition 4.1** Any continuous PWL function can be expressed as a difference of two convex PWL functions.

**Proposition 4.2** Any continuous function can be approximated arbitrarily well on a compact domain by a continuous PWL function.

**Theorem 4.3** Any continuous function can be approximated arbitrarily well on a compact domain by a maxout network with two maxout hidden units.

## 5. Benchmark results

## 6. Comparison to rectifiers

## 7. Model averaging

## 8. Optimization

----------------------------------------------------------------------------------------------------

## References

* Goodfellow, Ian, et al. "Maxout networks." *International conference on machine learning*. PMLR, 2013.

## Further Reading

* Dropout
