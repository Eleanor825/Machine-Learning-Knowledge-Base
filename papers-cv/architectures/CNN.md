<span style="font-family:monospace">

# Papers in Computer Vision - CNN Architectures

count: 59

## Unclassified

* [Deconvolutional networks](https://ieeexplore.ieee.org/document/5539957)
    * Title: Deconvolutional networks
    * Year: 05 Aug `2010`
    * Author: Matthew D. Zeiler
    * Abstract: Building robust low and mid-level image representations, beyond edge primitives, is a long-standing goal in vision. Many existing feature detectors spatially pool edge information which destroys cues such as edge intersections, parallelism and symmetry. We present a learning framework where features that capture these mid-level cues spontaneously emerge from image data. Our approach is based on the convolutional decomposition of images under a sparsity constraint and is totally unsupervised. By building a hierarchy of such decompositions we can learn rich feature sets that are a robust image representation for both the analysis and synthesis of images.
* [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
    * Title: ImageNet Classification with Deep Convolutional Neural Networks
    * Year: `2012`
    * Author: Alex Krizhevsky
    * Abstract: We trained a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7\% and 18.9\% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of convolutional nets. To reduce overfitting in the globally connected layers we employed a new regularization method that proved to be very effective.
* [DeCAF](https://arxiv.org/abs/1310.1531)
    * Title: DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition
    * Year: 06 Oct `2013`
    * Author: Jeff Donahue
    * Abstract: We evaluate whether features extracted from the activation of a deep convolutional network trained in a fully supervised fashion on a large, fixed set of object recognition tasks can be re-purposed to novel generic tasks. Our generic tasks may differ significantly from the originally trained tasks and there may be insufficient labeled or unlabeled data to conventionally train or adapt a deep architecture to the new tasks. We investigate and visualize the semantic clustering of deep convolutional features with respect to a variety of such tasks, including scene recognition, domain adaptation, and fine-grained recognition challenges. We compare the efficacy of relying on various network levels to define a fixed feature, and report novel results that significantly outperform the state-of-the-art on several important vision challenges. We are releasing DeCAF, an open-source implementation of these deep convolutional activation features, along with all associated network parameters to enable vision researchers to be able to conduct experimentation with deep representations across a range of visual concept learning paradigms.
* [Understanding Deep Architectures using a Recursive Convolutional Network](https://arxiv.org/abs/1312.1847)
    * Title: Understanding Deep Architectures using a Recursive Convolutional Network
    * Year: 06 Dec `2013`
    * Author: David Eigen
    * Abstract: A key challenge in designing convolutional network models is sizing them appropriately. Many factors are involved in these decisions, including number of layers, feature maps, kernel sizes, etc. Complicating this further is the fact that each of these influence not only the numbers and dimensions of the activation units, but also the total number of parameters. In this paper we focus on assessing the independent contributions of three of these linked variables: The numbers of layers, feature maps, and parameters. To accomplish this, we employ a recursive convolutional network whose weights are tied between layers; this allows us to vary each of the three factors in a controlled setting. We find that while increasing the numbers of layers and parameters each have clear benefit, the number of feature maps (and hence dimensionality of the representation) appears ancillary, and finds most of its benefit through the introduction of more weights. Our results (i) empirically confirm the notion that adding layers alone increases computational power, within the context of convolutional layers, and (ii) suggest that precise sizing of convolutional feature map dimensions is itself of little concern; more attention should be paid to the number of parameters in these layers instead.
* [VGG](https://arxiv.org/abs/1409.1556)
    * Title: Very Deep Convolutional Networks for Large-Scale Image Recognition
    * Year: 04 Sep `2014`
    * Author: Karen Simonyan
    * Abstract: In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localization and classification tracks respectively. We also show that our representations generalize well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.
* [FractalNet](https://arxiv.org/abs/1605.07648)
    * Title: FractalNet: Ultra-Deep Neural Networks without Residuals
    * Year: 24 May `2016`
    * Author: Gustav Larsson
* [DFN](https://arxiv.org/abs/1605.07716)
    * Title: Deeply-Fused Nets
    * Year: 25 May `2016`
    * Author: Jingdong Wang
* [SENet](https://arxiv.org/abs/1709.01507)
    * Title: Squeeze-and-Excitation Networks
    * Year: 05 Sep `2017`
    * Author: Jie Hu
    * Abstract: The central building block of convolutional neural networks (CNNs) is the convolution operator, which enables networks to construct informative features by fusing both spatial and channel-wise information within local receptive fields at each layer. A broad range of prior research has investigated the spatial component of this relationship, seeking to strengthen the representational power of a CNN by enhancing the quality of spatial encodings throughout its feature hierarchy. In this work, we focus instead on the channel relationship and propose a novel architectural unit, which we term the "Squeeze-and-Excitation" (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels. We show that these blocks can be stacked together to form SENet architectures that generalize extremely effectively across different datasets. We further demonstrate that SE blocks bring significant improvements in performance for existing state-of-the-art CNNs at slight additional computational cost. Squeeze-and-Excitation Networks formed the foundation of our ILSVRC 2017 classification submission which won first place and reduced the top-5 error to 2.251%, surpassing the winning entry of 2016 by a relative improvement of ~25%. Models and code are available at [this https URL](https://github.com/hujie-frank/SENet).
* [ESPNet](https://arxiv.org/abs/1803.06815)
    * Title: ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
    * Year: 19 Mar `2018`
    * Author: Sachin Mehta
    * Abstract: We introduce a fast and efficient convolutional neural network, ESPNet, for semantic segmentation of high resolution images under resource constraints. ESPNet is based on a new convolutional module, efficient spatial pyramid (ESP), which is efficient in terms of computation, memory, and power. ESPNet is 22 times faster (on a standard GPU) and 180 times smaller than the state-of-the-art semantic segmentation network PSPNet, while its category-wise accuracy is only 8% less. We evaluated ESPNet on a variety of semantic segmentation datasets including Cityscapes, PASCAL VOC, and a breast biopsy whole slide image dataset. Under the same constraints on memory and computation, ESPNet outperforms all the current efficient CNN networks such as MobileNet, ShuffleNet, and ENet on both standard metrics and our newly introduced performance metrics that measure efficiency on edge devices. Our network can process high resolution images at a rate of 112 and 9 frames per second on a standard GPU and edge device, respectively.
* [HRNet](https://arxiv.org/abs/1908.07919)
    * Title: Deep High-Resolution Representation Learning for Visual Recognition
    * Year: 20 Aug `2019`
    * Author: Jingdong Wang
    * Abstract: High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions \emph{in series} (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams \emph{in parallel}; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. We show the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems. All the codes are available at [this https URL](https://github.com/HRNet).
* [ConvNeXt](https://arxiv.org/abs/2201.03545)
    * Title: A ConvNet for the 2020s
    * Year: 10 Jan `2022`
    * Author: Zhuang Liu

## Reformulations of the Connections between Network Layers

> Highway networks introduced a gating mechanism to regulate the flow of information along shortcut connections. (SENet, 2017)

> Following these works, there have been further reformulations oof the connections between network layers [DPN], [DenseNets], which show promising improvements to the learning and representational properties of deep networks. (SENet, 2017)

> By revealing the equivalence of the state-of-the-art Residual Network (ResNet) and Densely Convolutional Network (DenseNet) within the HORNN framework, we find that ResNet enables feature re-usage while DenseNet enables new features exploration which are both important for learning good representations. (DPN, 2017)

> Although the width of the densely connected path increases linearly as it goes deeper, causing the number of parameters to grow quadratically, DenseNet provides higher parameter efficiency compared with ResNet. (DPN, 2017)

* ResNet, see below.
* Identity mappings, see below.
* [Highway Networks](https://arxiv.org/abs/1505.00387)
    * Title: Highway Networks
    * Year: 03 May `2015`
    * Author: Rupesh Kumar Srivastava
    * Abstract: There is plenty of theoretical and empirical evidence that depth of neural networks is a crucial ingredient for their success. However, network training becomes more difficult with increasing depth and training of very deep networks remains an open problem. In this extended abstract, we introduce a new architecture designed to ease gradient-based training of very deep networks. We refer to networks with this architecture as highway networks, since they allow unimpeded information flow across several layers on "information highways". The architecture is characterized by the use of gating units which learn to regulate the flow of information through a network. Highway networks with hundreds of layers can be trained directly using stochastic gradient descent and with a variety of activation functions, opening up the possibility of studying extremely deep and efficient architectures.
* [Training Very Deep Networks](https://arxiv.org/abs/1507.06228)
    * Title: Training Very Deep Networks
    * Year: 22 Jul `2015`
    * Author: Rupesh Kumar Srivastava
    * Abstract: Theoretical and empirical evidence indicates that the depth of neural networks is crucial for their success. However, training becomes more difficult as depth increases, and training of very deep networks remains an open problem. Here we introduce a new architecture designed to overcome this. Our so-called highway networks allow unimpeded information flow across many layers on information highways. They are inspired by Long Short-Term Memory recurrent networks and use adaptive gating units to regulate the information flow. Even with hundreds of layers, highway networks can be trained directly through simple gradient descent. This enables the study of extremely deep and efficient architectures.
* [DenseNet](https://arxiv.org/abs/1608.06993)
    * Title: Densely Connected Convolutional Networks
    * Year: 25 Aug `2016`
    * Author: Gao Huang
    * Abstract: Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance. Code and pre-trained models are available at [this https URL](https://github.com/liuzhuang13/DenseNet).
* [Dual Path Networks (DPN)](https://arxiv.org/abs/1707.01629)
    * Title: Dual Path Networks
    * Year: 06 Jul `2017`
    * Author: Yunpeng Chen
    * Abstract: In this work, we present a simple, highly efficient and modularized Dual Path Network (DPN) for image classification which presents a new topology of connection paths internally. By revealing the equivalence of the state-of-the-art Residual Network (ResNet) and Densely Convolutional Network (DenseNet) within the HORNN framework, we find that ResNet enables feature re-usage while DenseNet enables new features exploration which are both important for learning good representations. To enjoy the benefits from both path topologies, our proposed Dual Path Network shares common features while maintaining the flexibility to explore new features through dual path architectures. Extensive experiments on three benchmark datasets, ImagNet-1k, Places365 and PASCAL VOC, clearly demonstrate superior performance of the proposed DPN over state-of-the-arts. In particular, on the ImagNet-1k dataset, a shallow DPN surpasses the best ResNeXt-101(64x4d) with 26% smaller model size, 25% less computational cost and 8% lower memory consumption, and a deeper DPN (DPN-131) further pushes the state-of-the-art single model performance with about 2 times faster training speed. Experiments on the Places365 large-scale scene dataset, PASCAL VOC detection dataset, and PASCAL VOC segmentation dataset also demonstrate its consistently better performance than DenseNet, ResNet and the latest ResNeXt model over various applications.

## Residual Networks

> Deep Residual Network (ResNet) is one of the first works that successfully adopt skip connections, where each micro-block, .k.a. residual function, is associated with a skip connection, called residual path. The residual path element-wisely adds the input features to the output of the same micro-block, making it a residual unit. Depending on the inner structure design of the micro-block, the residual network has developed into a family  of various architectures, including WRN, Inception-resnet, and ResNeXt. (DPN, 2017)

* [ResNet](https://arxiv.org/abs/1512.03385)
    * Title: Deep Residual Learning for Image Recognition
    * Year: 10 Dec `2015`
    * Author: Kaiming He
* [ResNet with Pre-Activation](https://arxiv.org/abs/1603.05027)
    * Title: Identity Mappings in Deep Residual Networks
    * Year: 16 Mar `2016`
    * Author: Kaiming He
* [Generalized ResNet](https://arxiv.org/abs/1603.08029)
    * Title: Resnet in Resnet: Generalizing Residual Architectures
    * Year: 25 Mar `2016`
    * Author: Sasha Targ
* [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431)
    * Title: Residual Networks Behave Like Ensembles of Relatively Shallow Networks
    * Year: 20 May `2016`
    * Author: Andreas Veit
    * Abstract: In this work we propose a novel interpretation of residual networks showing that they can be seen as a collection of many paths of differing length. Moreover, residual networks seem to enable very deep networks by leveraging only the short paths during training. To support this observation, we rewrite residual networks as an explicit collection of paths. Unlike traditional models, paths through residual networks vary in length. Further, a lesion study reveals that these paths show ensemble-like behavior in the sense that they do not strongly depend on each other. Finally, and most surprising, most paths are shorter than one might expect, and only the short paths are needed during training, as longer paths do not contribute any gradient. For example, most of the gradient in a residual network with 110 layers comes from paths that are only 10-34 layers deep. Our results reveal one of the key characteristics that seem to enable the training of very deep networks: Residual networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of very deep networks.
* [Wide Residual Networks (WRN)](https://arxiv.org/abs/1605.07146)
    * Title: Wide Residual Networks
    * Year: 23 May `2016`
    * Author: Sergey Zagoruyko
* [ResNeXt](https://arxiv.org/abs/1611.05431)
    * Title: Aggregated Residual Transformations for Deep Neural Networks
    * Year: 16 Nov `2016`
    * Author: Saining Xie
    * Abstract: We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call "cardinality" (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online.
* [Collective Residual Unit (CRU)](https://arxiv.org/abs/1703.02180)
    * Title: Sharing Residual Units Through Collective Tensor Factorization in Deep Neural Networks
    * Year: 07 Mar `2017`
    * Author: Chen Yunpeng
    * Abstract: Residual units are wildly used for alleviating optimization difficulties when building deep neural networks. However, the performance gain does not well compensate the model size increase, indicating low parameter efficiency in these residual units. In this work, we first revisit the residual function in several variations of residual units and demonstrate that these residual functions can actually be explained with a unified framework based on generalized block term decomposition. Then, based on the new explanation, we propose a new architecture, Collective Residual Unit (CRU), which enhances the parameter efficiency of deep neural networks through collective tensor factorization. CRU enables knowledge sharing across different residual units using shared factors. Experimental results show that our proposed CRU Network demonstrates outstanding parameter efficiency, achieving comparable classification performance to ResNet-200 with the model size of ResNet-50. By building a deeper network using CRU, we can achieve state-of-the-art single model classification accuracy on ImageNet-1k and Places365-Standard benchmark datasets. (Code and trained models are available on GitHub)

## Recurrent Networks

* [Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex](https://arxiv.org/abs/1604.03640)
    * Title: Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex
    * Year: 13 Apr `2016`
    * Author: Qianli Liao
    * Abstract: We discuss relations between Residual Networks (ResNet), Recurrent Neural Networks (RNNs) and the primate visual cortex. We begin with the observation that a special type of shallow RNN is exactly equivalent to a very deep ResNet with weight sharing among the layers. A direct implementation of such a RNN, although having orders of magnitude fewer parameters, leads to a performance similar to the corresponding ResNet. We propose 1) a generalization of both RNN and ResNet architectures and 2) the conjecture that a class of moderately deep RNNs is a biologically-plausible model of the ventral stream in visual cortex. We demonstrate the effectiveness of the architectures by testing them on the CIFAR-10 and ImageNet dataset.
* [HORNN](https://arxiv.org/abs/1605.00064)
    * Title: Higher Order Recurrent Neural Networks
    * Year: 30 Apr `2016`
    * Author: Rohollah Soltani
    * Abstract: In this paper, we study novel neural network structures to better model long term dependency in sequential data. We propose to use more memory units to keep track of more preceding states in recurrent neural networks (RNNs), which are all recurrently fed to the hidden layers as feedback through different weighted paths. By extending the popular recurrent structure in RNNs, we provide the models with better short-term memory mechanism to learn long term dependency in sequences. Analogous to digital filters in signal processing, we call these structures as higher order RNNs (HORNNs). Similar to RNNs, HORNNs can also be learned using the back-propagation through time method. HORNNs are generally applicable to a variety of sequence modelling tasks. In this work, we have examined HORNNs for the language modeling task using two popular data sets, namely the Penn Treebank (PTB) and English text8 data sets. Experimental results have shown that the proposed HORNNs yield the state-of-the-art performance on both data sets, significantly outperforming the regular RNNs as well as the popular LSTMs.

## Grouped Convolutions

> Grouped convolutions have proven to be a popular approach for increasing the cardinality of learned transformations [Deep Roots], [ResNeXt]. (SENet, 2017)

* [Deep Roots](https://arxiv.org/abs/1605.06489)
    * Title: Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups
    * Year: 20 May `2016`
    * Author: Yani Ioannou
    * Abstract: We propose a new method for creating computationally efficient and compact convolutional neural networks (CNNs) using a novel sparse connection structure that resembles a tree root. This allows a significant reduction in computational cost and number of parameters compared to state-of-the-art deep CNNs, without compromising accuracy, by exploiting the sparsity of inter-layer filter dependencies. We validate our approach by using it to train more efficient variants of state-of-the-art CNN architectures, evaluated on the CIFAR10 and ILSVRC datasets. Our results show similar or higher accuracy than the baseline architectures with much less computation, as measured by CPU and GPU timings. For example, for ResNet 50, our model has 40% fewer parameters, 45% fewer floating point operations, and is 31% (12%) faster on a CPU (GPU). For the deeper ResNet 200 our model has 25% fewer floating point operations and 44% fewer parameters, while maintaining state-of-the-art accuracy. For GoogLeNet, our model has 7% fewer parameters and is 21% (16%) faster on a CPU (GPU).
* ResNeXt, see above.
* Inception-v1, Inception-v2, Inception-v3, Inception-v4, see below.

## Inception Networks (Multi-Branch Networks)

> The Inception models have evolved over time, but an important common property is a *split-transform-merge* strategy. In an Inception module, the input is split into a few lower-dimensional embeddings (by $1 \times 1$ convolutions), transformed by a set of specialized filters ($3 \times 3$, $5 \times 5$, etc.), and merged by concatenation. It can be shown that the solution space of this architecture is a strict subspace of the solution space of a single large layer (e.g., $5 \times 5$) operating on a high-dimensional embedding. The split-transform-merge behavior of Inception modules is expected to approach the representational power of large and dense layers, but at a considerably lower computational complexity. (ResNeXt, 2016)

> The Inception models are successful multi-branch architectures where each branch is carefully customized. ResNets can be thought of as two-branch networks where one branch is the identity mapping. Deep neural decision forests are tree-patterned multi-branch networks with learned splitting functions. (ResNeXt, 2016)

> More flexible compositions of operators can be achieved with multi-branch convolutions [Inception-v1], [Inception-v2], [Inception-v3], [Inception-v4], which can be viewed as a natural extension of the grouping operator. (SENet, 2017)

* [Inception-v1/GoogLeNet](https://arxiv.org/abs/1409.4842)
    * Title: Going Deeper with Convolutions
    * Year: 17 Sep `2014`
    * Author: Christian Szegedy
    * Abstract: We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.
* [Inception-v2/Batch Normalization](https://arxiv.org/abs/1502.03167)
    * Title: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    * Year: 11 Feb `2015`
    * Author: Sergey Ioffe
    * Abstract: Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.
* [Inception-v3](https://arxiv.org/abs/1512.00567)
    * Title: Rethinking the Inception Architecture for Computer Vision
    * Year: 02 Dec `2015`
    * Author: Christian Szegedy
    * Abstract: Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set.
* [Inception-v4/Inception-ResNet](https://arxiv.org/abs/1602.07261)
    * Title: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    * Year: 23 Feb `2016`
    * Author: Christian Szegedy
    * Abstract: Very deep convolutional networks have been central to the largest advances in image recognition performance in recent years. One example is the Inception architecture that has been shown to achieve very good performance at relatively low computational cost. Recently, the introduction of residual connections in conjunction with a more traditional architecture has yielded state-of-the-art performance in the 2015 ILSVRC challenge; its performance was similar to the latest generation Inception-v3 network. This raises the question of whether there are any benefit in combining the Inception architecture with residual connections. Here we give clear empirical evidence that training with residual connections accelerates the training of Inception networks significantly. There is also some evidence of residual Inception networks outperforming similarly expensive Inception networks without residual connections by a thin margin. We also present several new streamlined architectures for both residual and non-residual Inception networks. These variations improve the single-frame recognition performance on the ILSVRC 2012 classification task significantly. We further demonstrate how proper activation scaling stabilizes the training of very wide residual Inception networks. With an ensemble of three residual and one Inception-v4, we achieve 3.08 percent top-5 error on the test set of the ImageNet classification (CLS) challenge.
* [PolyNet](https://arxiv.org/abs/1611.05725)
    * Title: PolyNet: A Pursuit of Structural Diversity in Very Deep Networks
    * Year: 17 Nov `2016`
    * Author: Xingcheng Zhang
    * Abstract: A number of studies have shown that increasing the depth or width of convolutional networks is a rewarding approach to improve the performance of image recognition. In our study, however, we observed difficulties along both directions. On one hand, the pursuit for very deep networks is met with a diminishing return and increased training difficulty; on the other hand, widening a network would result in a quadratic growth in both computational cost and memory demand. These difficulties motivate us to explore structural diversity in designing deep networks, a new dimension beyond just depth and width. Specifically, we present a new family of modules, namely the PolyInception, which can be flexibly inserted in isolation or in a composition as replacements of different parts of a network. Choosing PolyInception modules with the guidance of architectural efficiency can improve the expressive power while preserving comparable computational cost. The Very Deep PolyNet, designed following this direction, demonstrates substantial improvements over the state-of-the-art on the ILSVRC 2012 benchmark. Compared to Inception-ResNet-v2, it reduces the top-5 validation error on single crops from 4.9% to 4.25%, and that on multi-crops from 3.7% to 3.45%.

## Deformable Convolutions

* [Deformable ConvNets v1](https://arxiv.org/abs/1703.06211)
    * Title: Deformable Convolutional Networks
    * Year: 17 Mar `2017`
    * Author: Jifeng Dai
    * Abstract: Convolutional neural networks (CNNs) are inherently limited to model geometric transformations due to the fixed geometric structures in its building modules. In this work, we introduce two new modules to enhance the transformation modeling capacity of CNNs, namely, deformable convolution and deformable RoI pooling. Both are based on the idea of augmenting the spatial sampling locations in the modules with additional offsets and learning the offsets from target tasks, without additional supervision. The new modules can readily replace their plain counterparts in existing CNNs and can be easily trained end-to-end by standard back-propagation, giving rise to deformable convolutional networks. Extensive experiments validate the effectiveness of our approach on sophisticated vision tasks of object detection and semantic segmentation. The code would be released.
* [Deformable ConvNets v2](https://arxiv.org/abs/1811.11168)
    * Title: Deformable ConvNets v2: More Deformable, Better Results
    * Year: 27 Nov `2018`
    * Author: Xizhou Zhu
    * Abstract: The superior performance of Deformable Convolutional Networks arises from its ability to adapt to the geometric variations of objects. Through an examination of its adaptive behavior, we observe that while the spatial support for its neural features conforms more closely than regular ConvNets to object structure, this support may nevertheless extend well beyond the region of interest, causing features to be influenced by irrelevant image content. To address this problem, we present a reformulation of Deformable ConvNets that improves its ability to focus on pertinent image regions, through increased modeling power and stronger training. The modeling power is enhanced through a more comprehensive integration of deformable convolution within the network, and by introducing a modulation mechanism that expands the scope of deformation modeling. To effectively harness this enriched modeling capability, we guide network training via a proposed feature mimicking scheme that helps the network to learn features that reflect the object focus and classification power of R-CNN features. With the proposed contributions, this new version of Deformable ConvNets yields significant performance gains over the original model and produces leading results on the COCO benchmark for object detection and instance segmentation.

## Light Weight Networks

* [SqueezeNet](https://arxiv.org/abs/1602.07360)
    * Title: SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
    * Year: 24 Feb `2016`
    * Author: Forrest N. Iandola
    * Abstract: Recent research on deep neural networks has focused primarily on improving accuracy. For a given accuracy level, it is typically possible to identify multiple DNN architectures that achieve that accuracy level. With equivalent accuracy, smaller DNN architectures offer at least three advantages: (1) Smaller DNNs require less communication across servers during distributed training. (2) Smaller DNNs require less bandwidth to export a new model from the cloud to an autonomous car. (3) Smaller DNNs are more feasible to deploy on FPGAs and other hardware with limited memory. To provide all of these advantages, we propose a small DNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet). The SqueezeNet architecture is available for download here: [this https URL](https://github.com/DeepScale/SqueezeNet).
* [ShuffleNet V1](https://arxiv.org/abs/1707.01083)
    * Title: ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    * Year: 04 Jul `2017`
    * Author: Xiangyu Zhang
    * Abstract: We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet on ImageNet classification task, under the computation budget of 40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves ~13x actual speedup over AlexNet while maintaining comparable accuracy.
* [ShiftNet](https://arxiv.org/abs/1711.08141)
    * Title: Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions
    * Year: 22 Nov `2017`
    * Author: Bichen Wu
    * Abstract: Neural networks rely on convolutions to aggregate spatial information. However, spatial convolutions are expensive in terms of model size and computation, both of which grow quadratically with respect to kernel size. In this paper, we present a parameter-free, FLOP-free "shift" operation as an alternative to spatial convolutions. We fuse shifts and point-wise convolutions to construct end-to-end trainable shift-based modules, with a hyperparameter characterizing the tradeoff between accuracy and efficiency. To demonstrate the operation's efficacy, we replace ResNet's 3x3 convolutions with shift-based modules for improved CIFAR10 and CIFAR100 accuracy using 60% fewer parameters; we additionally demonstrate the operation's resilience to parameter reduction on ImageNet, outperforming ResNet family members. We finally show the shift operation's applicability across domains, achieving strong performance with fewer parameters on classification, face verification and style transfer.
* [CondenseNet](https://arxiv.org/abs/1711.09224)
    * Title: CondenseNet: An Efficient DenseNet using Learned Group Convolutions
    * Year: 25 Nov `2017`
    * Author: Gao Huang
    * Abstract: Deep neural networks are increasingly used on mobile devices, where computational resources are limited. In this paper we develop CondenseNet, a novel network architecture with unprecedented efficiency. It combines dense connectivity with a novel module called learned group convolution. The dense connectivity facilitates feature re-use in the network, whereas learned group convolutions remove connections between layers for which this feature re-use is superfluous. At test time, our model can be implemented using standard group convolutions, allowing for efficient computation in practice. Our experiments show that CondenseNets are far more efficient than state-of-the-art compact convolutional networks such as MobileNets and ShuffleNets.

## Light-weight CNNs (MobileViT, 2021)

> The basic building layer in CNNs is a standard convolutional layer. Because this layer is computationally expensive, several factorization-based methods have been proposed to make it light-weight and mobile-friendly. Of these, separable convolutions of Chollet (2017) have gained interest, and are widely used across state-of-the-art light-weight CNNs for mobile vision tasks, including MobileNets, ShuffleNetv2, ESPNetv2, MixNet, and MNASNet. These light-weight CNNs are versatile and easy to train. For example, these networks can easily replace the heavy-weight backbones (e.g., ResNet) in existing task-specific models (e.g., DeepLabv3) to reduce the network size and improve latency. Despite these benefits, one major drawback of these methods is that they are spatially local. (MobileViT, 2021)

* [Flattened Networks](https://arxiv.org/abs/1412.5474)
    * Title: Flattened Convolutional Neural Networks for Feedforward Acceleration
    * Year: 17 Dec `2014`
    * Author: Jonghoon Jin
    * Abstract: We present flattened convolutional neural networks that are designed for fast feedforward execution. The redundancy of the parameters, especially weights of the convolutional filters in convolutional neural networks has been extensively studied and different heuristics have been proposed to construct a low rank basis of the filters after training. In this work, we train flattened networks that consist of consecutive sequence of one-dimensional filters across all directions in 3D space to obtain comparable performance as conventional convolutional networks. We tested flattened model on different datasets and found that the flattened layer can effectively substitute for the 3D filters without loss of accuracy. The flattened convolution pipelines provide around two times speed-up during feedforward pass compared to the baseline model due to the significant reduction of learning parameters. Furthermore, the proposed method does not require efforts in manual tuning or post processing once the model is trained.
* Xception: Deep Learning with Depthwise Separable Convolutions
* [DiCENet](https://arxiv.org/abs/1906.03516)
    * Title: DiCENet: Dimension-wise Convolutions for Efficient Networks
    * Year: 08 Jun `2019`
    * Author: Sachin Mehta
    * Abstract: We introduce a novel and generic convolutional unit, DiCE unit, that is built using dimension-wise convolutions and dimension-wise fusion. The dimension-wise convolutions apply light-weight convolutional filtering across each dimension of the input tensor while dimension-wise fusion efficiently combines these dimension-wise representations; allowing the DiCE unit to efficiently encode spatial and channel-wise information contained in the input tensor. The DiCE unit is simple and can be seamlessly integrated with any architecture to improve its efficiency and performance. Compared to depth-wise separable convolutions, the DiCE unit shows significant improvements across different architectures. When DiCE units are stacked to build the DiCENet model, we observe significant improvements over state-of-the-art models across various computer vision tasks including image classification, object detection, and semantic segmentation. On the ImageNet dataset, the DiCENet delivers 2-4% higher accuracy than state-of-the-art manually designed models (e.g., MobileNetv2 and ShuffleNetv2). Also, DiCENet generalizes better to tasks (e.g., object detection) that are often used in resource-constrained devices in comparison to state-of-the-art separable convolution-based efficient networks, including neural search-based methods (e.g., MobileNetv3 and MixNet. Our source code in PyTorch is open-source and is available at [this https URL](https://github.com/sacmehta/EdgeNets/).

* MobileNetV1, MobileNetV2, MobileNetV3.
* [ShuffleNet V2](https://arxiv.org/abs/1807.11164)
    * Title: ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    * Year: 30 Jul `2018`
    * Author: Ningning Ma
    * Abstract: Currently, the neural network architecture design is mostly guided by the \emph{indirect} metric of computation complexity, i.e., FLOPs. However, the \emph{direct} metric, e.g., speed, also depends on the other factors such as memory access cost and platform characterics. Thus, this work proposes to evaluate the direct metric on the target platform, beyond only considering FLOPs. Based on a series of controlled experiments, this work derives several practical \emph{guidelines} for efficient network design. Accordingly, a new architecture is presented, called \emph{ShuffleNet V2}. Comprehensive ablation experiments verify that our model is the state-of-the-art in terms of speed and accuracy tradeoff.
* [ESPNetv2](https://arxiv.org/abs/1811.11431)
    * Title: ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network
    * Year: 28 Nov `2018`
    * Author: Sachin Mehta
    * Abstract: We introduce a light-weight, power efficient, and general purpose convolutional neural network, ESPNetv2, for modeling visual and sequential data. Our network uses group point-wise and depth-wise dilated separable convolutions to learn representations from a large effective receptive field with fewer FLOPs and parameters. The performance of our network is evaluated on four different tasks: (1) object classification, (2) semantic segmentation, (3) object detection, and (4) language modeling. Experiments on these tasks, including image classification on the ImageNet and language modeling on the PenTree bank dataset, demonstrate the superior performance of our method over the state-of-the-art methods. Our network outperforms ESPNet by 4-5% and has 2-4x fewer FLOPs on the PASCAL VOC and the Cityscapes dataset. Compared to YOLOv2 on the MS-COCO object detection, ESPNetv2 delivers 4.4% higher accuracy with 6x fewer FLOPs. Our experiments show that ESPNetv2 is much more power efficient than existing state-of-the-art efficient methods including ShuffleNets and MobileNets. Our code is open-source and available at [this https URL](https://github.com/sacmehta/ESPNetv2).
* [Mixconv](https://arxiv.org/abs/1907.09595)
    * Title: Mixconv: Mixed depthwise convolutional kernels
    * Year: 22 Jul `2019`
    * Author: Mingxing Tan
    * Abstract: Depthwise convolution is becoming increasingly popular in modern efficient ConvNets, but its kernel size is often overlooked. In this paper, we systematically study the impact of different kernel sizes, and observe that combining the benefits of multiple kernel sizes can lead to better accuracy and efficiency. Based on this observation, we propose a new mixed depthwise convolution (MixConv), which naturally mixes up multiple kernel sizes in a single convolution. As a simple drop-in replacement of vanilla depthwise convolution, our MixConv improves the accuracy and efficiency for existing MobileNets on both ImageNet classification and COCO object detection. To demonstrate the effectiveness of MixConv, we integrate it into AutoML search space and develop a new family of models, named as MixNets, which outperform previous mobile models including MobileNetV2 [20] (ImageNet top-1 accuracy +4.2%), ShuffleNetV2 [16] (+3.5%), MnasNet [26] (+1.3%), ProxylessNAS [2] (+2.2%), and FBNet [27] (+2.0%). In particular, our MixNet-L achieves a new state-of-the-art 78.9% ImageNet top-1 accuracy under typical mobile settings (<600M FLOPS). Code is at [this https URL](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet).
* [MnasNet](https://arxiv.org/abs/1807.11626)
    * Title: MnasNet: Platform-Aware Neural Architecture Search for Mobile
    * Year: 31 Jul `2018`
    * Author: Mingxing Tan
    * Abstract: Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider. In this paper, we propose an automated mobile neural architecture search (MNAS) approach, which explicitly incorporate model latency into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. Unlike previous work, where latency is considered via another, often inaccurate proxy (e.g., FLOPS), our approach directly measures real-world inference latency by executing the model on mobile phones. To further strike the right balance between flexibility and search space size, we propose a novel factorized hierarchical search space that encourages layer diversity throughout the network. Experimental results show that our approach consistently outperforms state-of-the-art mobile CNN models across multiple vision tasks. On the ImageNet classification task, our MnasNet achieves 75.2% top-1 accuracy with 78ms latency on a Pixel phone, which is 1.8x faster than MobileNetV2 [29] with 0.5% higher accuracy and 2.3x faster than NASNet [36] with 1.2% higher accuracy. Our MnasNet also achieves better mAP quality than MobileNets for COCO object detection. Code is at [this https URL](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet).

## Mobile Networks (3)

* [MobileNetV1](https://arxiv.org/abs/1704.04861)
    * Title: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    * Year: 17 Apr `2017`
    * Author: Andrew G. Howard
    * Abstract: We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.
* [MobileNetV2](https://arxiv.org/abs/1801.04381)
    * Title: MobileNetV2: Inverted Residuals and Linear Bottlenecks
    * Year: 13 Jan `2018`
    * Author: Mark Sandler
    * Abstract: In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3. The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters
* [MobileNetV3](https://arxiv.org/abs/1905.02244)
    * Title: Searching for MobileNetV3
    * Year: 06 May `2019`
    * Author: Andrew Howard
    * Abstract: We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2\% more accurate on ImageNet classification while reducing latency by 15\% compared to MobileNetV2. MobileNetV3-Small is 4.6\% more accurate while reducing latency by 5\% compared to MobileNetV2. MobileNetV3-Large detection is 25\% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30\% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.

## Efficient Networks (2)

* [EfficientNetV1](https://arxiv.org/abs/1905.11946)
    * Title: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    * Year: 28 May `2019`
    * Author: Mingxing Tan
    * Abstract: Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet. To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at [this https URL](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).
* [EfficientNetV2](https://arxiv.org/abs/2104.00298)
    * Title: EfficientNetV2: Smaller Models and Faster Training
    * Year: 01 Apr `2021`
    * Author: Mingxing Tan
    * Abstract: This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. To develop this family of models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller. Our training can be further sped up by progressively increasing the image size during training, but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose to adaptively adjust regularization (e.g., dropout and data augmentation) as well, such that we can achieve both fast training and good accuracy. With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources. Code will be available at [this https URL](https://github.com/google/automl/tree/master/efficientnetv2).

## Compressing Convolutional Networks

* [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](https://arxiv.org/abs/1404.0736)
    * Title: Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation
    * Year: 02 Apr `2014`
    * Author: Emily Denton
    * Abstract: We present techniques for speeding up the test-time evaluation of large convolutional networks, designed for object recognition tasks. These models deliver impressive accuracy but each image evaluation requires millions of floating point operations, making their deployment on smartphones and Internet-scale clusters problematic. The computation is dominated by the convolution operations in the lower layers of the model. We exploit the linear structure present within the convolutional filters to derive approximations that significantly reduce the required computation. Using large state-of-the-art models, we demonstrate we demonstrate speedups of convolutional layers on both CPU and GPU by a factor of 2x, while keeping the accuracy within 1% of the original model.
* [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530)
    * Title: Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications
    * Year: 20 Nov `2015`
    * Author: Yong-Deok Kim
    * Abstract: Although the latest high-end smartphone has powerful CPU and GPU, running deeper convolutional neural networks (CNNs) for complex tasks such as ImageNet classification on mobile devices is challenging. To deploy deep CNNs on mobile devices, we present a simple and effective scheme to compress the entire CNN, which we call one-shot whole network compression. The proposed scheme consists of three steps: (1) rank selection with variational Bayesian matrix factorization, (2) Tucker decomposition on kernel tensor, and (3) fine-tuning to recover accumulated loss of accuracy, and each step can be easily implemented using publicly available tools. We demonstrate the effectiveness of the proposed scheme by testing the performance of various compressed CNNs (AlexNet, VGGS, GoogLeNet, and VGG-16) on the smartphone. Significant reductions in model size, runtime, and energy consumption are obtained, at the cost of small loss in accuracy. In addition, we address the important implementation level issue on 1?1 convolution, which is a key operation of inception module of GoogLeNet as well as CNNs compressed by our proposed scheme.
* Speeding up with low rank expansions
* Deep roots

## Cross-Channel Correlations

> In prior work, cross channel correlations are typically mapped as new combinations of features, either independently of spatial structure [speeding up], [Xception] or jointly by using standard convolutional filters [Network in Network] with $1 \times 1$ convolutions. (SENet, 2017)

* [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)
    * Title: Speeding up Convolutional Neural Networks with Low Rank Expansions
    * Year: 15 May `2014`
    * Author: Max Jaderberg
    * Abstract: The focus of this paper is speeding up the evaluation of convolutional neural networks. While delivering impressive results across a range of computer vision and machine learning tasks, these networks are computationally demanding, limiting their deployability. Convolutional layers generally consume the bulk of the processing time, and so in this work we present two simple schemes for drastically speeding up these layers. This is achieved by exploiting cross-channel or filter redundancy to construct a low rank basis of filters that are rank-1 in the spatial domain. Our methods are architecture agnostic, and can be easily applied to existing CPU and GPU convolutional frameworks for tuneable speedup performance. We demonstrate this with a real world network designed for scene text character recognition, showing a possible 2.5x speedup with no loss in accuracy, and 4.5x speedup with less than 1% drop in accuracy, still achieving state-of-the-art on standard benchmarks.
* [Xception Networks](https://arxiv.org/abs/1610.02357)
    * Title: Xception: Deep Learning with Depthwise Separable Convolutions
    * Year: 07 Oct `2016`
    * Author: François Chollet
    * Abstract: We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.
* NIN, see `mlp.md`.

## Low-Rank Approximations

* [Deformable kernels for early vision](https://ieeexplore.ieee.org/document/391394)
    * Title: Deformable kernels for early vision
    * Year: May `1995`
    * Author: P. Perona
    * Abstract: Early vision algorithms often have a first stage of linear-filtering that 'extracts' from the image information at multiple scales of resolution and multiple orientations. A common difficulty in the design and implementation of such schemes is that one feels compelled to discretize coarsely the space of scales and orientations in order to reduce computation and storage costs. A technique is presented that allows: 1) computing the best approximation of a given family using linear combinations of a small number of 'basis' functions; and 2) describing all finite-dimensional families, i.e., the families of filters for which a finite dimensional representation is possible with no error. The technique is based on singular value decomposition and may be applied to generating filters in arbitrary dimensions and subject to arbitrary deformations. The relevant functional analysis results are reviewed and precise conditions for the decomposition to be feasible are stated. Experimental results are presented that demonstrate the applicability of the technique to generating multiorientation multi-scale 2D edge-detection kernels. The implementation issues are also discussed.
* [A rank minimization heuristic with application to minimum order system approximation](https://ieeexplore.ieee.org/document/945730)
    * Title: A rank minimization heuristic with application to minimum order system approximation
    * Year: 07 August `2002`
    * Author: M. Fazel
    * Abstract: We describe a generalization of the trace heuristic that applies to general nonsymmetric, even non-square, matrices, and reduces to the trace heuristic when the matrix is positive semidefinite. The heuristic is to replace the (nonconvex) rank objective with the sum of the singular values of the matrix, which is the dual of the spectral norm. We show that this problem can be reduced to a semidefinite program, hence efficiently solved. To motivate the heuristic, we, show that the dual spectral norm is the convex envelope of the rank on the set of matrices with norm less than one. We demonstrate the method on the problem of minimum-order system approximation.
* [Decompositions of a Higher-Order Tensor in Block Terms—Part II: Definitions and Uniqueness](https://epubs.siam.org/doi/10.1137/070690729)
    * Title: Decompositions of a Higher-Order Tensor in Block Terms—Part II: Definitions and Uniqueness
    * Year: `2008`
    * Author: Lieven De Lathauwer
    * Abstract: In this paper we introduce a new class of tensor decompositions. Intuitively, we decompose a given tensor block into blocks of smaller size, where the size is characterized by a set of mode-n ranks. We study different types of such decompositions. For each type we derive conditions under which essential uniqueness is guaranteed. The parallel factor decomposition and Tucker's decomposition can be considered as special cases in the new framework. The paper sheds new light on fundamental aspects of tensor algebra.
* [Tensor Decompositions and Applications](https://epubs.siam.org/doi/10.1137/07070111X)
    * Title: Tensor Decompositions and Applications
    * Year: `2009`
    * Author: Tamara G. Kolda
    * Abstract: This survey provides an overview of higher-order tensor decompositions, their applications, and available software. A tensor is a multidimensional or N-way array. Decompositions of higher-order tensors (i.e., N-way arrays with $N \geq 3$) have applications in psycho-metrics, chemometrics, signal processing, numerical linear algebra, computer vision, numerical analysis, data mining, neuroscience, graph analysis, and elsewhere. Two particular tensor decompositions can be considered to be higher-order extensions of the matrix singular value decomposition: CANDECOMP/PARAFAC (CP) decomposes a tensor as a sum of rank-one tensors, and the Tucker decomposition is a higher-order form of principal component analysis. There are many other tensor decompositions, including INDSCAL, PARAFAC2, CANDELINC, DEDICOM, and PARATUCK2 as well as nonnegative variants of all of the above. The N-way Toolbox, Tensor Toolbox, and Multilinear Engine are examples of software packages for working with tensors.
* [Learning Separable Filters](https://ieeexplore.ieee.org/document/6619199)
    * Title: Learning Separable Filters
    * Year: 03 October `2013`
    * Author: Roberto Rigamonti
    * Abstract: Learning filters to produce sparse image representations in terms of over complete dictionaries has emerged as a powerful way to create image features for many different purposes. Unfortunately, these filters are usually both numerous and non-separable, making their use computationally expensive. In this paper, we show that such filters can be computed as linear combinations of a smaller number of separable ones, thus greatly reducing the computational complexity at no cost in terms of performance. This makes filter learning approaches practical even for large images or 3D volumes, and we show that we significantly outperform state-of-the-art methods on the linear structure extraction task, in terms of both accuracy and speed. Moreover, our approach is general and can be used on generic filter banks to reduce the complexity of the convolutions.
* [Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition](https://arxiv.org/abs/1412.6553)
    * Title: Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition
    * Year: 19 Dec `2014`
    * Author: Vadim Lebedev
    * Abstract: We propose a simple two-step approach for speeding up convolution layers within large convolutional neural networks based on tensor decomposition and discriminative fine-tuning. Given a layer, we use non-linear least squares to compute a low-rank CP-decomposition of the 4D convolution kernel tensor into a sum of a small number of rank-one tensors. At the second step, this decomposition is used to replace the original convolutional layer with a sequence of four convolutional layers with small kernels. After such replacement, the entire network is fine-tuned on the training data using standard backpropagation process. We evaluate this approach on two CNNs and show that it is competitive with previous approaches, leading to higher obtained CPU speedups at the cost of lower accuracy drops for the smaller of the two networks. Thus, for the 36-class character classification CNN, our approach obtains a 8.5x CPU speedup of the whole network with only minor accuracy drop (1% from 91% to 90%). For the standard ImageNet architecture (AlexNet), the approach speeds up the second convolution layer by a factor of 4x at the cost of 1% increase of the overall top-5 classification error.
* [Training CNNs with Low-Rank Filters for Efficient Image Classification](https://arxiv.org/abs/1511.06744)
    * Title: Training CNNs with Low-Rank Filters for Efficient Image Classification
    * Year: 20 Nov `2015`
    * Author: Yani Ioannou
    * Abstract: We propose a new method for creating computationally efficient convolutional neural networks (CNNs) by using low-rank representations of convolutional filters. Rather than approximating filters in previously-trained networks with more efficient versions, we learn a set of small basis filters from scratch; during training, the network learns to combine these basis filters into more complex filters that are discriminative for image classification. To train such networks, a novel weight initialization scheme is used. This allows effective initialization of connection weights in convolutional layers composed of groups of differently-shaped filters. We validate our approach by applying it to several existing CNN architectures and training these networks from scratch using the CIFAR, ILSVRC and MIT Places datasets. Our results show similar or higher accuracy than conventional CNNs with much less compute. Applying our method to an improved version of VGG-11 network using global max-pooling, we achieve comparable validation accuracy using 41% less compute and only 24% of the original VGG-11 model parameters; another variant of our method gives a 1 percentage point increase in accuracy over our improved VGG-11 model, giving a top-5 center-crop validation accuracy of 89.7% while reducing computation by 16% relative to the original VGG-11 model. Applying our method to the GoogLeNet architecture for ILSVRC, we achieved comparable accuracy with 26% less compute and 41% fewer model parameters. Applying our method to a near state-of-the-art network for CIFAR, we achieved comparable accuracy with 46% less compute and 55% fewer parameters.
* Speeding up with low rank expansions
* Exploiting Linear Structure

## Regularization Techniques

* [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
    * Title: Improving neural networks by preventing co-adaptation of feature detectors
    * Year: 03 Jul `2012`
    * Author: Geoffrey E. Hinton
    * Abstract: When a large feedforward neural network is trained on a small training set, it typically performs poorly on held-out test data. This "overfitting" is greatly reduced by randomly omitting half of the feature detectors on each training case. This prevents complex co-adaptations in which a feature detector is only helpful in the context of several other specific feature detectors. Instead, each neuron learns to detect a feature that is generally helpful for producing the correct answer given the combinatorially large variety of internal contexts in which it must operate. Random "dropout" gives big improvements on many benchmark tasks and sets new records for speech and object recognition.
* [DropConnect](https://proceedings.mlr.press/v28/wan13.html)
    * Title: Regularization of Neural Networks using DropConnect
    * Year: `2013`
    * Author: Li Wan
    * Abstract: We introduce DropConnect, a generalization of DropOut, for regularizing large fully-connected layers within neural networks. When training with Dropout, a randomly selected subset of activations are set to zero within each layer. DropConnect instead sets a randomly selected subset of weights within the network to zero. Each unit thus receives input from a random subset of units in the previous layer. We derive a bound on the generalization performance of both Dropout and DropConnect. We then evaluate DropConnect on a range of datasets, comparing to Dropout, and show state-of-the-art results on several image recognition benchmarks can be obtained by aggregating multiple DropConnect-trained models.
* [Stochastic Pooling](https://arxiv.org/abs/1301.3557)
    * Title: Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
    * Year: 16 Jan `2013`
    * Author: Matthew D. Zeiler
    * Abstract: We introduce a simple and effective method for regularizing large convolutional neural networks. We replace the conventional deterministic pooling operations with a stochastic procedure, randomly picking the activation within each pooling region according to a multinomial distribution, given by the activities within the pooling region. The approach is hyper-parameter free and can be combined with other regularization approaches, such as dropout and data augmentation. We achieve state-of-the-art performance on four image datasets, relative to other approaches that do not utilize data augmentation.
* [Maxout](https://arxiv.org/abs/1302.4389)
    * Title: Maxout Networks
    * Year: 18 Feb `2013`
    * Author: Ian J. Goodfellow
    * Abstract: We consider the problem of designing models to leverage a recently introduced approximate model averaging technique called dropout. We define a simple new model called maxout (so named because its output is the max of a set of inputs, and because it is a natural companion to dropout) designed to both facilitate optimization by dropout and improve the accuracy of dropout's fast approximate model averaging technique. We empirically verify that the model successfully accomplishes both of these tasks. We use maxout and dropout to demonstrate state of the art classification performance on four benchmark datasets: MNIST, CIFAR-10, CIFAR-100, and SVHN.
* [Dropout](https://dl.acm.org/doi/10.5555/2627435.2670313)
    * Title: Dropout: a simple way to prevent neural networks from overfitting
    * Year: 01 Jan `2014`
    * Author: Nitish Srivastava
    * Abstract: Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. During training, dropout samples from an exponential number of different "thinned" networks. At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods. We show that dropout improves the performance of neural networks on supervised learning tasks in vision, speech recognition, document classification and computational biology, obtaining state-of-the-art results on many benchmark data sets.
* [DropIn](https://arxiv.org/abs/1511.06951)
    * Title: Gradual DropIn of Layers to Train Very Deep Neural Networks
    * Year: 22 Nov `2015`
    * Author: Leslie N. Smith
    * Abstract: We introduce the concept of dynamically growing a neural network during training. In particular, an untrainable deep network starts as a trainable shallow network and newly added layers are slowly, organically added during training, thereby increasing the network's depth. This is accomplished by a new layer, which we call DropIn. The DropIn layer starts by passing the output from a previous layer (effectively skipping over the newly added layers), then increasingly including units from the new layers for both feedforward and backpropagation. We show that deep networks, which are untrainable with conventional methods, will converge with DropIn layers interspersed in the architecture. In addition, we demonstrate that DropIn provides regularization during training in an analogous way as dropout. Experiments are described with the MNIST dataset and various expanded LeNet architectures, CIFAR-10 dataset with its architecture expanded from 3 to 11 layers, and on the ImageNet dataset with the AlexNet architecture expanded to 13 layers and the VGG 16-layer architecture.
* [Stochastic Depth](https://arxiv.org/abs/1603.09382)
    * Title: Deep Networks with Stochastic Depth
    * Year: 30 Mar `2016`
    * Author: Gao Huang
    * Abstract: Very deep convolutional networks with hundreds of layers have led to significant reductions in error on competitive benchmarks. Although the unmatched expressiveness of the many layers can be highly desirable at test time, training very deep networks comes with its own set of challenges. The gradients can vanish, the forward flow often diminishes, and the training time can be painfully slow. To address these problems, we propose stochastic depth, a training procedure that enables the seemingly contradictory setup to train short networks and use deep networks at test time. We start with very deep networks but during training, for each mini-batch, randomly drop a subset of layers and bypass them with the identity function. This simple approach complements the recent success of residual networks. It reduces training time substantially and improves the test error significantly on almost all data sets that we used for evaluation. With stochastic depth we can increase the depth of residual networks even beyond 1200 layers and still yield meaningful improvements in test error (4.91% on CIFAR-10).
