<span style="font-family:monospace">

# Papers in Training Methodologies

count: 6

## Normalization

* [Batch Normalization](https://arxiv.org/abs/1502.03167)
    * Title: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    * Year: 11 Feb `2015`
    * Author: Sergey Ioffe
    * Abstract: Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.
* (21 Jul 2016) [Layer Normalization](https://arxiv.org/abs/1607.06450)
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
