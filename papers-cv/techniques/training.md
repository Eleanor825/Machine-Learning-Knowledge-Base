<span style="font-family:monospace">

# Papers in Training Methodologies

count: 8

## Vanishing Gradients

* [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
    * Title: Understanding the difficulty of training deep feedforward neural networks
    * Year: May 2010
    * Author: Xavier Glorot
    * Abstract: Whereas before 2006 it appears that deep multi-layer neural networks were not successfully trained, since then several algorithms have been shown to successfully train them, with experimental results showing the superiority of deeper vs less deep architectures. All these experimental results were obtained with new initialization or training mechanisms. Our objective here is to understand better why standard gradient descent from random initialization is doing so poorly with deep neural networks, to better understand these recent relative successes and help design better algorithms in the future. We first observe the influence of the non-linear activations functions. We find that the logistic sigmoid activation is unsuited for deep networks with random initialization because of its mean value, which can drive especially the top hidden layer into saturation. Surprisingly, we find that saturated units can move out of saturation by themselves, albeit slowly, and explaining the plateaus sometimes seen when training neural networks. We find that a new non-linearity that saturates less can often be beneficial. Finally, we study how activations and gradients vary across layers and during training, with the idea that training may be more difficult when the singular values of the Jacobian associated with each layer are far from 1. Based on these considerations, we propose a new initialization scheme that brings substantially faster convergence.
* [DSN](https://arxiv.org/abs/1409.5185)
    * Title: Deeply-Supervised Nets
    * Year: 18 Sep 2014
    * Author: Chen-Yu Lee
    * Abstract: Our proposed deeply-supervised nets (DSN) method simultaneously minimizes classification error while making the learning process of hidden layers direct and transparent. We make an attempt to boost the classification performance by studying a new formulation in deep networks. Three aspects in convolutional neural networks (CNN) style architectures are being looked at: (1) transparency of the intermediate layers to the overall classification; (2) discriminativeness and robustness of learned features, especially in the early layers; (3) effectiveness in training due to the presence of the exploding and vanishing gradients. We introduce "companion objective" to the individual hidden layers, in addition to the overall objective at the output layer (a different strategy to layer-wise pre-training). We extend techniques from stochastic gradient methods to analyze our algorithm. The advantage of our method is evident and our experimental result on benchmark datasets shows significant performance gain over existing methods (e.g. all state-of-the-art results on MNIST, CIFAR-10, CIFAR-100, and SVHN).
* [FitNets](https://arxiv.org/abs/1412.6550)
    * Title: FitNets: Hints for Thin Deep Nets
    * Year: 19 Dec 2014
    * Author: FitNets: Hints for Thin Deep Nets
    * Abstract: While depth tends to improve network performances, it also makes gradient-based training more difficult since deeper networks tend to be more non-linear. The recently proposed knowledge distillation approach is aimed at obtaining small and fast-to-execute models, and it has shown that a student network could imitate the soft output of a larger teacher network or ensemble of networks. In this paper, we extend this idea to allow the training of a student that is deeper and thinner than the teacher, using not only the outputs but also the intermediate representations learned by the teacher as hints to improve the training process and final performance of the student. Because the student intermediate hidden layer will generally be smaller than the teacher's intermediate hidden layer, additional parameters are introduced to map the student hidden layer to the prediction of the teacher hidden layer. This allows one to train deeper students that can generalize better or run faster, a trade-off that is controlled by the chosen student capacity. For example, on CIFAR-10, a deep student network with almost 10.4 times less parameters outperforms a larger, state-of-the-art teacher network.

## Normalization

* [Layer Normalization](https://arxiv.org/abs/1607.06450)
    * Title: Layer Normalization
    * Year: 21 Jul `2016`
    * Author: Jimmy Lei Ba
    * Abstract: Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.
* (27 Jul 2016) [Instance Normalization](https://arxiv.org/abs/1607.08022)
* (22 Mar 2018) [Group Normalization](https://arxiv.org/abs/1803.08494)

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
