<span style="font-family:monospace">

# Papers in Training Methodologies

count: 12

## Vanishing Gradients

* [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
    * Title: Understanding the difficulty of training deep feedforward neural networks
    * Year: May 2010
    * Author: Xavier Glorot
    * Abstract: Whereas before 2006 it appears that deep multi-layer neural networks were not successfully trained, since then several algorithms have been shown to successfully train them, with experimental results showing the superiority of deeper vs less deep architectures. All these experimental results were obtained with new initialization or training mechanisms. Our objective here is to understand better why standard gradient descent from random initialization is doing so poorly with deep neural networks, to better understand these recent relative successes and help design better algorithms in the future. We first observe the influence of the non-linear activations functions. We find that the logistic sigmoid activation is unsuited for deep networks with random initialization because of its mean value, which can drive especially the top hidden layer into saturation. Surprisingly, we find that saturated units can move out of saturation by themselves, albeit slowly, and explaining the plateaus sometimes seen when training neural networks. We find that a new non-linearity that saturates less can often be beneficial. Finally, we study how activations and gradients vary across layers and during training, with the idea that training may be more difficult when the singular values of the Jacobian associated with each layer are far from 1. Based on these considerations, we propose a new initialization scheme that brings substantially faster convergence.

## Normalization

> Applying either weight normalization or batch normalization using expected statistics is equivalent to have a different parameterization of the original feed-forward neural network. Re-parameterization in the ReLU network was studied in the Path-normalized SGD. Our proposed layer normalization method, however, is not a re-parameterization of the original neural network. The layer normalized model, thus, has different invariance properties than the other methods. (Layer Normalization, 2016)

* Batch Normalization
* [Layer Normalization](https://arxiv.org/abs/1607.06450)
    * Title: Layer Normalization
    * Year: 21 Jul `2016`
    * Author: Jimmy Lei Ba
    * Abstract: Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.
* [Instance Normalization](https://arxiv.org/abs/1607.08022)
    * Title: Instance Normalization: The Missing Ingredient for Fast Stylization
    * Year: 27 Jul `2016`
    * Author: Dmitry Ulyanov
    * Abstract: It this paper we revisit the fast stylization method introduced in Ulyanov et. al. (2016). We show how a small change in the stylization architecture results in a significant qualitative improvement in the generated images. The change is limited to swapping batch normalization with instance normalization, and to apply the latter both at training and testing times. The resulting method can be used to train high-performance architectures for real-time image generation. The code will is made available on github at [this https URL](https://github.com/DmitryUlyanov/texture_nets). Full paper can be found at [arXiv:1701.02096](https://arxiv.org/abs/1701.02096).
* [Group Normalization](https://arxiv.org/abs/1803.08494)
    * Title: Group Normalization
    * Year: 22 Mar `2018`
    * Author: Yuxin Wu
    * Abstract: Batch Normalization (BN) is a milestone technique in the development of deep learning, enabling various networks to train. However, normalizing along the batch dimension introduces problems --- BN's error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation. This limits BN's usage for training larger models and transferring features to computer vision tasks including detection, segmentation, and video, which require small batches constrained by memory consumption. In this paper, we present Group Normalization (GN) as a simple alternative to BN. GN divides the channels into groups and computes within each group the mean and variance for normalization. GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes. On ResNet-50 trained in ImageNet, GN has 10.6% lower error than its BN counterpart when using a batch size of 2; when using typical batch sizes, GN is comparably good with BN and outperforms other normalization variants. Moreover, GN can be naturally transferred from pre-training to fine-tuning. GN can outperform its BN-based counterparts for object detection and segmentation in COCO, and for video classification in Kinetics, showing that GN can effectively replace the powerful BN in a variety of tasks. GN can be easily implemented by a few lines of code in modern libraries.
* [Weight Normalization](https://arxiv.org/abs/1602.07868)
    * Title: Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    * Year: 25 Feb `2016`
    * Author: Tim Salimans
    * Abstract: We present weight normalization: a reparameterization of the weight vectors in a neural network that decouples the length of those weight vectors from their direction. By reparameterizing the weights in this way we improve the conditioning of the optimization problem and we speed up convergence of stochastic gradient descent. Our reparameterization is inspired by batch normalization but does not introduce any dependencies between the examples in a minibatch. This means that our method can also be applied successfully to recurrent models such as LSTMs and to noise-sensitive applications such as deep reinforcement learning or generative models, for which batch normalization is less well suited. Although our method is much simpler, it still provides much of the speed-up of full batch normalization. In addition, the computational overhead of our method is lower, permitting more optimization steps to be taken in the same amount of time. We demonstrate the usefulness of our method on applications in supervised image recognition, generative modelling, and deep reinforcement learning.
* [Path-SGD: Path-Normalized Optimization in Deep Neural Networks](https://arxiv.org/abs/1506.02617)
    * Title: Path-SGD: Path-Normalized Optimization in Deep Neural Networks
    * Year: 08 Jun `2015`
    * Author: Behnam Neyshabur
    * Abstract: We revisit the choice of SGD for training deep neural networks by reconsidering the appropriate geometry in which to optimize the weights. We argue for a geometry invariant to rescaling of weights that does not affect the output of the network, and suggest Path-SGD, which is an approximate steepest descent method with respect to a path-wise regularizer related to max-norm regularization. Path-SGD is easy and efficient to implement and leads to empirical gains over SGD and AdaGrad.

## Label Smoothing

* [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)
    * Title: When Does Label Smoothing Help?
    * Year: 06 Jun `2019`
    * Author: Rafael Müller
    * Abstract: The generalization and learning speed of a multi-class neural network can often be significantly improved by using soft targets that are a weighted average of the hard targets and the uniform distribution over labels. Smoothing the labels in this way prevents the network from becoming over-confident and label smoothing has been used in many state-of-the-art models, including image classification, language translation and speech recognition. Despite its widespread use, label smoothing is still poorly understood. Here we show empirically that in addition to improving generalization, label smoothing improves model calibration which can significantly improve beam-search. However, we also observe that if a teacher network is trained with label smoothing, knowledge distillation into a student network is much less effective. To explain these observations, we visualize how label smoothing changes the representations learned by the penultimate layer of the network. We show that label smoothing encourages the representations of training examples from the same class to group in tight clusters. This results in loss of information in the logits about resemblances between instances of different classes, which is necessary for distillation, but does not hurt generalization or calibration of the model's predictions.
* [Delving Deep into Label Smoothing](https://arxiv.org/abs/2011.12562)
    * Title: Delving Deep into Label Smoothing
    * Year: 25 Nov `2020`
    * Author: Chang-Bin Zhang
    * Abstract: Label smoothing is an effective regularization tool for deep neural networks (DNNs), which generates soft labels by applying a weighted average between the uniform distribution and the hard label. It is often used to reduce the overfitting problem of training DNNs and further improve classification performance. In this paper, we aim to investigate how to generate more reliable soft labels. We present an Online Label Smoothing (OLS) strategy, which generates soft labels based on the statistics of the model prediction for the target category. The proposed OLS constructs a more reasonable probability distribution between the target categories and non-target categories to supervise DNNs. Experiments demonstrate that based on the same classification models, the proposed approach can effectively improve the classification performance on CIFAR-100, ImageNet, and fine-grained datasets. Additionally, the proposed method can significantly improve the robustness of DNN models to noisy labels compared to current label smoothing approaches.

## Regularization

* [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    * Title: Decoupled Weight Decay Regularization
    * Year: 14 Nov `2017`
    * Authors: Ilya Loshchilov, Frank Hutter
    * Abstract: $L_{2}$ regularization and weight decay regularization are equivalent for standard stochastic gradient descent (when rescaled by the learning rate), but as we demonstrate this is \emph{not} the case for adaptive gradient algorithms, such as Adam. While common implementations of these algorithms employ $L_{2}$ regularization (often calling it "weight decay" in what may be misleading due to the inequivalence we expose), we propose a simple modification to recover the original formulation of weight decay regularization by \emph{decoupling} the weight decay from the optimization steps taken w.r.t. the loss function. We provide empirical evidence that our proposed modification (i) decouples the optimal choice of weight decay factor from the setting of the learning rate for both standard SGD and Adam and (ii) substantially improves Adam's generalization performance, allowing it to compete with SGD with momentum on image classification datasets (on which it was previously typically outperformed by the latter). Our proposed decoupled weight decay has already been adopted by many researchers, and the community has implemented it in TensorFlow and PyTorch; the complete source code for our experiments is available at [this https URL](https://github.com/loshchil/AdamW-and-SGDW).

## Others

* [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
    * Title: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
    * Year: 08 Jun `2017`
    * Authors: Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He
    * Abstract: Deep learning thrives with large neural networks and large datasets. However, larger networks and larger datasets result in longer training times that impede research and development progress. Distributed synchronous SGD offers a potential solution to this problem by dividing SGD minibatches over a pool of parallel workers. Yet to make this scheme efficient, the per-worker workload must be large, which implies nontrivial growth in the SGD minibatch size. In this paper, we empirically show that on the ImageNet dataset large minibatches cause optimization difficulties, but when these are addressed the trained networks exhibit good generalization. Specifically, we show no loss of accuracy when training with large minibatch sizes up to 8192 images. To achieve this result, we adopt a hyper-parameter-free linear scaling rule for adjusting learning rates as a function of minibatch size and develop a new warmup scheme that overcomes optimization challenges early in training. With these simple techniques, our Caffe2-based system trains ResNet-50 with a minibatch size of 8192 on 256 GPUs in one hour, while matching small minibatch accuracy. Using commodity hardware, our implementation achieves ~90% scaling efficiency when moving from 8 to 256 GPUs. Our findings enable training visual recognition models on internet-scale data with high efficiency.
