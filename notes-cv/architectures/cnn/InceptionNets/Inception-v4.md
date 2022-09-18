# [Notes][Vision][CNN] Inception-v4/Inception-ResNet

* url: https://arxiv.org/abs/1602.07261
* Title: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
* Year: 23 Feb `2016`
* Authors: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
* Institutions: [Google Inc.]
* Abstract: Very deep convolutional networks have been central to the largest advances in image recognition performance in recent years. One example is the Inception architecture that has been shown to achieve very good performance at relatively low computational cost. Recently, the introduction of residual connections in conjunction with a more traditional architecture has yielded state-of-the-art performance in the 2015 ILSVRC challenge; its performance was similar to the latest generation Inception-v3 network. This raises the question of whether there are any benefit in combining the Inception architecture with residual connections. Here we give clear empirical evidence that training with residual connections accelerates the training of Inception networks significantly. There is also some evidence of residual Inception networks outperforming similarly expensive Inception networks without residual connections by a thin margin. We also present several new streamlined architectures for both residual and non-residual Inception networks. These variations improve the single-frame recognition performance on the ILSVRC 2012 classification task significantly. We further demonstrate how proper activation scaling stabilizes the training of very wide residual Inception networks. With an ensemble of three residual and one Inception-v4, we achieve 3.08 percent top-5 error on the test set of the ImageNet classification (CLS) challenge.

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* New models: Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2.
* New ideas: Scaling of the Residuals.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> In this work we study the combination of the two most recent ideas: Residual connections introduced by He et al. in  [5] and the latest revised version of the Inception architecture [15].

> Since Inception networks tend to be very deep, it is natural to replace the filter concatenation stage of the Inception architecture with residual connections. This would allow Inception to reap all the benefits of the residual approach while retaining its computational efficiency.

> Besides a straightforward integration, we have also studied whether Inception itself can be made more efficient by making it deeper and wider. For that purpose, we designed a new version named Inception-v4 which has a more `uniform simplified architecture` and more inception modules than Inception-v3.

## 2. Related Work

> Residual connection were introduced by He et al. in [5] in which they give convincing theoretical and practical evidence for the advantages of utilizing additive merging of signals both for image recognition, and especially for object detection. The authors argue that residual connections are inherently necessary for training very deep convolutional models.

> Our findings do not seem to support this view, at least for image recognition. However it might require more measurement points with deeper architectures to understand the true extent of beneficial aspects offered by residual connections. In the experimental section we demonstrate that it is not very difficult to train competitive very deep networks without utilizing residual connections. However the use of residual connections seems to improve the training speed greatly, which is alone a great argument for their use.

## 3. Architectural Choices

### 3.1. Pure Inception blocks

> Our older Inception models used to be trained in a partitioned manner, where each replica was partitioned into a multiple sub-networks in order to be able to fit the whole model in memory.

> However, the Inception architecture is highly tunable, meaning that there are a lot of possible changes to the number of filters in the various layers that do not affect the quality of the fully trained network.

> Not simplifying earlier choices resulted in networks that looked more complicated that they needed to be. In our newer experiments, for Inception-v4 we decided to shed this unnecessary baggage and made uniform choices for the Inception blocks for each grid size.

### 3.2. Residual Inception Blocks

> For the residual versions of the Inception networks, we use cheaper Inception blocks than the original Inception.

### 3.3. Scaling of the Residuals

> We found that scaling down the residuals before adding them to the previous layer activation seemed to stabilize the training. In general we picked some scaling factors between 0.1 and 0.3 to scale the residuals before their being added to the accumulated layer activations (cf. Figure 20).

> Even where the scaling was not strictly necessary, it never seemed to harm the final accuracy, but it helped to stabilize the training.

## 4. Training Methodology

## 5. Experimental Results

## 6. Conclusions

----------------------------------------------------------------------------------------------------

## References

* Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." *Thirty-first AAAI conference on artificial intelligence*. 2017.

## Further Reading

* [4] R-CNN
* [5] ResNet
* [6] Inception-v2/Batch Normalization
* [8] AlexNet
* [9] Network In Network (NIN)
* [10] Fully Convolutional Networks (FCN)
* [12] VGGNet
* [14] [Inception-v1/GoogLeNet](https://zhuanlan.zhihu.com/p/564141144)
* [15] Inception-v3
* [17] DeepPose
