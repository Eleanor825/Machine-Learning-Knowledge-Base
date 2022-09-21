#! https://zhuanlan.zhihu.com/p/565285454
# [Notes][Vision][CNN] AlexNet

* url: https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
* Title: ImageNet Classification with Deep Convolutional Neural Networks
* Year: `2012`
* Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
* Institution: [University of Toronto]
* Abstract: We trained a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7\% and 18.9\% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of convolutional nets. To reduce overfitting in the globally connected layers we employed a new regularization method that proved to be very effective.

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Preprocessing: subtract the mean activity over the training set from each pixel.
* Activation: ReLU activation.
* Training on Multiple GPUs: the GPUs communicate only in certain layers.
* Local Response Normalization
* Results: top-1 error of 37.5% and top-5 error of 17.0% on ILSVRC-2010.

----------------------------------------------------------------------------------------------------

## 1 Introduction

## 2 The Dataset

> ImageNet consists of variable-resolution images, while our system requires a constant input dimensionality.

> Therefore, we down-sampled the images to a fixed resolution of 256x256. Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then cropped out the central 256x256 patch from the resulting image.

## 3 The Architecture

### 3.1 ReLU Nonlinearity

> The standard way to model a neuron's output $f$ as a function of its input $x$ is with $f(x) = \tanh(x)$ or $f(x) = (1 + e^{-x})^{-1}$. In terms of training time with gradient descent, these `saturating` nonlinearities are much slower than the `non-saturating` nonlinearity $f(x) = \max(0, x)$.

> Deep convolutional neural networks with ReLUs train several times faster than their equivalents with tanh units.

### 3.2 Training on Multiple GPUs

> Current GPUs are particularly well-suited to cross-GPU parallelization, as they are able to read from and write to one another's memory directly, without going through host machine memory.

> The parallelization scheme that we employ essentially puts half of the kernels (or neurons) on each GPU, with one additional trick: the GPUs communicate only in certain layers.

> The resultant architecture is somewhat similar to that of the "columnar" CNN employed by Cire¸san et al. [5], except that our columns are not independent (see Figure 2).

### 3.3 Local Response Normalization

> ReLUs have the desirable property that they do not require input normalization to prevent them from saturating.

Notations:
* Let $a_{x,y}^{i}$ denote the activity of a neuron computed by applying kernel $i$ at position $(x, y)$ and then applying the ReLU nonlinearity.
* Let $b_{x,y}^{i}$ denote the response-normalized activity.

Then
$$b_{x,y}^{i} := a_{x,y}^{i} / (k + \alpha\sum_{j=r}^{s}(a_{x,y}^{j})^{2})^{\beta}$$
where $r := \max(0, i-n/2)$ and $s := \min(N-1, i+n/2)$ are lower and upper bounds of the summation
and the constants $k$, $n$, $\alpha$, $\beta$ are hyperparameters.

> We used $k = 2$, $n = 5$, $\alpha = 10^{-4}$, and $\beta = 0.75$.

> We applied this normalization after applying the ReLU nonlinearity in certain layers (see Section 3.5).

> This scheme bears some resemblance to the local contrast normalization scheme of Jarrett et al. [11], but ours would be more correctly termed "brightness normalization", since we do not subtract the mean activity.

### 3.4 Overlapping Pooling

> Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map. Traditionally, the neighborhoods summarized by adjacent pooling units do not overlap (e.g., [17, 11, 4]).

> We generally observe during training that models with overlapping pooling find it slightly more difficult to overfit.

### 3.5 Overall Architecture

> The `ReLU non-linearity` is applied to the output of every convolutional and fully-connected layer.
> `Response-normalization` layers follow the first and second convolutional layers.
> `Max-pooling` layers, of the kind described in Section 3.4, follow both response-normalization layers as well as the fifth convolutional layer.

## 4 Reducing Overfitting

### 4.1 Data Augmentation

First Technique

> The first form of data augmentation consists of generating image translations and horizontal reflections. We do this by extracting random 224x224 patches (and their horizontal reflections) from the 256×256 images and training our network on these extracted patches.This increases the size of our training set by a factor of 2048, though the resulting training examples are, of course, highly interdependent.

> At test time, the network makes a prediction by extracting five 224x224 patches (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the predictions made by the network's softmax layer on the ten patches.

Second Technique

> The second form of data augmentation consists of altering the intensities of the RGB channels in training images.

> This scheme approximately captures an important property of natural images, namely, that object identity is invariant to changes in the intensity and color of the illumination.

### 4.2 Dropout

> The neurons which are "dropped out" in this way do not contribute to the forward pass and do not participate in backpropagation. So every time an input is presented, the neural network samples a different architecture, but all these architectures share weights.

> This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.

> At test time, we use all the neurons but multiply their outputs by 0.5, which is a reasonable approximation to taking the geometric mean of the predictive distributions produced by the exponentially-many dropout networks.

## 5 Details of learning

> We used an equal learning rate for all layers, which we adjusted manually throughout training. The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate.

## 6 Results

## 7 Discussion

> Our results show that a large, deep convolutional neural network is capable of achieving record-breaking results on a highly challenging dataset using purely supervised learning.

> It is notable that our network's performance degrades if a single convolutional layer is removed.

----------------------------------------------------------------------------------------------------

## References

* Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." *Communications of the ACM* 60.6 (2017): 84-90.

## Further Reading

