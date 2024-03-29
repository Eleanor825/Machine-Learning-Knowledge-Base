# [Papers][Vision] CNN Architectures <!-- omit in toc -->

count=149

## Table of Contents <!-- omit in toc -->

- [Basics](#basics)
- [Deconvolution](#deconvolution)
- [Research on Network Topology](#research-on-network-topology)
  - [Reformulations of the Connections between Network Layers](#reformulations-of-the-connections-between-network-layers)
  - [Ladder Networks](#ladder-networks)
  - [Residual Networks Family](#residual-networks-family)
  - [Dense Topology Framework](#dense-topology-framework)
  - [Grouped Convolutions](#grouped-convolutions)
  - [InceptionNet Series](#inceptionnet-series)
  - [Multi-Column Networks](#multi-column-networks)
- [Research on Convolutional Kernels](#research-on-convolutional-kernels)
  - [Deformable Kernels](#deformable-kernels)
  - [Dilated Kernels](#dilated-kernels)
  - [Adaptive Kernels](#adaptive-kernels)
  - [Cross-Channel Correlations](#cross-channel-correlations)
- [Research on Attention Mechanism](#research-on-attention-mechanism)
  - [Survey](#survey)
  - [Unknown](#unknown)
  - [Spatial Attention](#spatial-attention)
  - [(2022, SegNeXt) Channel Attention (count=3)](#2022-segnext-channel-attention-count3)
- [Light-Weight Networks](#light-weight-networks)
  - [Light-Weight Networks (Others)](#light-weight-networks-others)
  - [(2017, MobileNetV1) Light-Weight Networks (count=7)](#2017-mobilenetv1-light-weight-networks-count7)
  - [(2021, MobileViTv1) Light-Weight Networks (count=3+5)](#2021-mobilevitv1-light-weight-networks-count35)
  - [MobileNet Series (count=3+2)](#mobilenet-series-count32)
  - [EfficientNets (count=3)](#efficientnets-count3)
  - [ShuffleNets (count=2)](#shufflenets-count2)
- [Information Bottleneck](#information-bottleneck)
- [Model Compression](#model-compression)
  - [(2016, RexNeXt) Compressing Convolutional Networks (count=4)](#2016-rexnext-compressing-convolutional-networks-count4)
  - [(ESPNetV1, 2018) Compressing Convolutional Networks (count=3)](#espnetv1-2018-compressing-convolutional-networks-count3)
  - [(ESPNetV2, 2018) Compressing Convolutional Networks (count=2)](#espnetv2-2018-compressing-convolutional-networks-count2)
  - [Low-bit Networks (2017, MobileNetV1) (count=3)](#low-bit-networks-2017-mobilenetv1-count3)
  - [Low-bit Networks (2018, ESPNetV1) (4)](#low-bit-networks-2018-espnetv1-4)
  - [Low-bit Networks (2018, ESPNetV2) (3)](#low-bit-networks-2018-espnetv2-3)
- [Low-Rank Approximations](#low-rank-approximations)
  - [Low-Rank Approximations](#low-rank-approximations-1)
  - [Low-Rank Approximations (2016, Deep Roots) (count=3+3)](#low-rank-approximations-2016-deep-roots-count33)
  - [Tensor Decomposition (ShuffleNet V2, 2018)](#tensor-decomposition-shufflenet-v2-2018)
- [Sparsity](#sparsity)
  - [(2018, ESPNetV1) (count=3)](#2018-espnetv1-count3)
  - [(2018, MobileNetV2) (count=1)](#2018-mobilenetv2-count1)
  - [Depth Completion](#depth-completion)
- [Expressive Power (2019, EfficientNetV1) (4)](#expressive-power-2019-efficientnetv1-4)
- [Reparameterization](#reparameterization)
- [Regularization Techniques](#regularization-techniques)
- [Data Augmentation](#data-augmentation)
- [Progressive Learning](#progressive-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Connectivity Learning (2018, MobileNetV2) (2)](#connectivity-learning-2018-mobilenetv2-2)
- [Unclassified (listed in to be read order)](#unclassified-listed-in-to-be-read-order)

----------------------------------------------------------------------------------------------------

## Basics

* [[Backpropagation Applied to Handwritten Zip Code Recognition](https://ieeexplore.ieee.org/document/6795724)]
    * Title: Backpropagation Applied to Handwritten Zip Code Recognition
    * Year: December `1989`
    * Authors: Y. LeCun; B. Boser; J. S. Denker; D. Henderson; R. E. Howard; W. Hubbard; L. D. Jackel
    * Abstract: The ability of learning networks to generalize can be greatly enhanced by providing constraints from the task domain. This paper demonstrates how such constraints can be integrated into a backpropagation network through the architecture of the network. This approach has been successfully applied to the recognition of handwritten zip code digits provided by the U.S. Postal Service. A single network learns the entire recognition operation, going from the normalized image of the character to the final classification.
* [[LeNet](https://ieeexplore.ieee.org/document/726791)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791)]
    * Title: Gradient-based learning applied to document recognition
    * Year: November `1998`
    * Authors: Y. Lecun; L. Bottou; Y. Bengio; P. Haffner
    * Abstract: Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique. Given an appropriate network architecture, gradient-based learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns, such as handwritten characters, with minimal preprocessing. This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task. Convolutional neural networks, which are specifically designed to deal with the variability of 2D shapes, are shown to outperform all other techniques. Real-life document recognition systems are composed of multiple modules including field extraction, segmentation recognition, and language modeling. A new learning paradigm, called graph transformer networks (GTN), allows such multimodule systems to be trained globally using gradient-based methods so as to minimize an overall performance measure. Two systems for online handwriting recognition are described. Experiments demonstrate the advantage of global training, and the flexibility of graph transformer networks. A graph transformer network for reading a bank cheque is also described. It uses convolutional neural network character recognizers combined with global training techniques to provide record accuracy on business and personal cheques. It is deployed commercially and reads several million cheques per day.
* [[AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)] <!-- printed -->
    [[pdf](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)]
    * Title: ImageNet Classification with Deep Convolutional Neural Networks
    * Year: `2012`
    * Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    * Institution: [University of Toronto]
    * Abstract: We trained a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7\% and 18.9\% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of convolutional nets. To reduce overfitting in the globally connected layers we employed a new regularization method that proved to be very effective.
    * Comments:
        * > (2017, ShuffleNetV1) The concept of group convolution, which was first introduced in AlexNet [21] for distributing the model over two GPUs, has been well demonstrated its effectiveness in ResNeXt [40].
* [[ZFNet](https://arxiv.org/abs/1311.2901)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1311.2901.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1311.2901/)]
    * Title: Visualizing and Understanding Convolutional Networks
    * Year: 12 Nov `2013`
    * Authors: Matthew D Zeiler, Rob Fergus
    * Institutions: [Dept. of Computer Science, Courant Institute, New York University]
    * Abstract: Large Convolutional Network models have recently demonstrated impressive classification performance on the ImageNet benchmark. However there is no clear understanding of why they perform so well, or how they might be improved. In this paper we address both issues. We introduce a novel visualization technique that gives insight into the function of intermediate feature layers and the operation of the classifier. We also perform an ablation study to discover the performance contribution from different model layers. This enables us to find model architectures that outperform Krizhevsky \etal on the ImageNet classification benchmark. We show our ImageNet model generalizes well to other datasets: when the softmax classifier is retrained, it convincingly beats the current state-of-the-art results on Caltech-101 and Caltech-256 datasets.
* [[VGGNet](https://arxiv.org/abs/1409.1556)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1409.1556.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1409.1556/)]
    * Title: Very Deep Convolutional Networks for Large-Scale Image Recognition
    * Year: 04 Sep `2014`
    * Authors: Karen Simonyan, Andrew Zisserman
    * Institutions: [Visual Geometry Group, Department of Engineering Science, University of Oxford]
    * Abstract: In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

## Deconvolution

* [[Deconvolutional networks](https://ieeexplore.ieee.org/document/5539957)] <!-- printed -->
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5539957)]
    * Title: Deconvolutional networks
    * Year: 05 Aug `2010`
    * Authors: Matthew D. Zeiler; Dilip Krishnan; Graham W. Taylor; Rob Fergus
    * Abstract: Building robust low and mid-level image representations, beyond edge primitives, is a long-standing goal in vision. Many existing feature detectors spatially pool edge information which destroys cues such as edge intersections, parallelism and symmetry. We present a learning framework where features that capture these mid-level cues spontaneously emerge from image data. Our approach is based on the convolutional decomposition of images under a sparsity constraint and is totally unsupervised. By building a hierarchy of such decompositions we can learn rich feature sets that are a robust image representation for both the analysis and synthesis of images.
* [[Adaptive Deconvolutional Networks](https://ieeexplore.ieee.org/document/6126474)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6126474)]
    * Title: Adaptive deconvolutional networks for mid and high level feature learning
    * Year: 12 January `2012`
    * Authors: Matthew D. Zeiler; Graham W. Taylor; Rob Fergus
    * Abstract: We present a hierarchical model that learns image decompositions via alternating layers of convolutional sparse coding and max pooling. When trained on natural images, the layers of our model capture image information in a variety of forms: low-level edges, mid-level edge junctions, high-level object parts and complete objects. To build our model we rely on a novel inference scheme that ensures each layer reconstructs the input, rather than just the output of the layer directly beneath, as is common with existing hierarchical approaches. This makes it possible to learn multiple layers of representation and we show models with 4 layers, trained on images from the Caltech-101 and 256 datasets. When combined with a standard classifier, features extracted from these models outperform SIFT, as well as representations from other feature learning methods.
    * Comments:
        * > One insight is that spatial information lost during max-pooling can in part be recovered by unpooling and deconvolution [36] providing a useful way to visualize input dependency in feed-forward models [35]. (LRR, 2016)

## Research on Network Topology

### Reformulations of the Connections between Network Layers

> Following these works, there have been further reformulations of the connections between network layers [DPN], [DenseNets], which show promising improvements to the learning and representational properties of deep networks. (SENet, 2017)

* [[Highway Networks](https://arxiv.org/abs/1505.00387)]
    [[pdf](https://arxiv.org/pdf/1505.00387.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1505.00387/)]
    * Title: Highway Networks
    * Year: 03 May `2015`
    * Authors: Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    * Institutions: The Swiss AI Lab IDSIA / USI / SUPSI
    * Abstract: There is plenty of theoretical and empirical evidence that depth of neural networks is a crucial ingredient for their success. However, network training becomes more difficult with increasing depth and training of very deep networks remains an open problem. In this extended abstract, we introduce a new architecture designed to ease gradient-based training of very deep networks. We refer to networks with this architecture as highway networks, since they allow unimpeded information flow across several layers on "information highways". The architecture is characterized by the use of gating units which learn to regulate the flow of information through a network. Highway networks with hundreds of layers can be trained directly using stochastic gradient descent and with a variety of activation functions, opening up the possibility of studying extremely deep and efficient architectures.
* [[Training Very Deep Networks](https://arxiv.org/abs/1507.06228)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1507.06228.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1507.06228/)]
    * Title: Training Very Deep Networks
    * Year: 22 Jul `2015`
    * Authors: Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    * Institutions: The Swiss AI Lab IDSIA / USI / SUPSI
    * Abstract: Theoretical and empirical evidence indicates that the depth of neural networks is crucial for their success. However, training becomes more difficult as depth increases, and training of very deep networks remains an open problem. Here we introduce a new architecture designed to overcome this. Our so-called highway networks allow unimpeded information flow across many layers on information highways. They are inspired by Long Short-Term Memory recurrent networks and use adaptive gating units to regulate the information flow. Even with hundreds of layers, highway networks can be trained directly through simple gradient descent. This enables the study of extremely deep and efficient architectures.
    * Comments:
        * > (2015, ResNet) Concurrent with our work, "highway networks" [42, 43] present shortcut connections with gating functions [15].
        * > (2016, DenseNet) ResNets [11] and Highway Networks [33] bypass signal from one layer to the next via identity connections.
        * > (2016, DenseNet) Highway Networks [33] were amongst the first architectures that provided a means to effectively train end-to-end networks with more than 100 layers. Using bypassing paths along with gating units, Highway Networks with hundreds of layers can be optimized with SGD effectively. The bypassing paths are presumed to be the key factor that eases the training of these very deep networks.
        * > (2017, SENet) Highway networks introduced a gating mechanism to regulate the flow of information along shortcut connections.
* [[Stochastic Depth](https://arxiv.org/abs/1603.09382)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1603.09382.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1603.09382/)]
    * Title: Deep Networks with Stochastic Depth
    * Year: 30 Mar `2016`
    * Authors: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
    * Abstract: Very deep convolutional networks with hundreds of layers have led to significant reductions in error on competitive benchmarks. Although the unmatched expressiveness of the many layers can be highly desirable at test time, training very deep networks comes with its own set of challenges. The gradients can vanish, the forward flow often diminishes, and the training time can be painfully slow. To address these problems, we propose stochastic depth, a training procedure that enables the seemingly contradictory setup to train short networks and use deep networks at test time. We start with very deep networks but during training, for each mini-batch, randomly drop a subset of layers and bypass them with the identity function. This simple approach complements the recent success of residual networks. It reduces training time substantially and improves the test error significantly on almost all data sets that we used for evaluation. With stochastic depth we can increase the depth of residual networks even beyond 1200 layers and still yield meaningful improvements in test error (4.91% on CIFAR-10).
    * Comments:
        * > (2016, DenseNet) Stochastic depth [13] shortens ResNets by randomly dropping layers during training to allow better information and gradient flow.
        * > (2016, DenseNet) Recent variations of ResNets [13] show that many layers contribute very little and can in fact be randomly dropped during training.
        * > (2016, DenseNet) Stochastic depth improves the training of deep residual networks by dropping layers randomly during training.
* [[FractalNet](https://arxiv.org/abs/1605.07648)]
    [[pdf](https://arxiv.org/pdf/1605.07648.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.07648/)]
    * Title: FractalNet: Ultra-Deep Neural Networks without Residuals
    * Year: 24 May `2016`
    * Authors: Gustav Larsson, Michael Maire, Gregory Shakhnarovich
    * Abstract: We introduce a design strategy for neural network macro-architecture based on self-similarity. Repeated application of a simple expansion rule generates deep networks whose structural layouts are precisely truncated fractals. These networks contain interacting subpaths of different lengths, but do not include any pass-through or residual connections; every internal signal is transformed by a filter and nonlinearity before being seen by subsequent layers. In experiments, fractal networks match the excellent performance of standard residual networks on both CIFAR and ImageNet classification tasks, thereby demonstrating that residual representations may not be fundamental to the success of extremely deep convolutional neural networks. Rather, the key may be the ability to transition, during training, from effectively shallow to deep. We note similarities with student-teacher behavior and develop drop-path, a natural extension of dropout, to regularize co-adaptation of subpaths in fractal architectures. Such regularization allows extraction of high-performance fixed-depth subnetworks. Additionally, fractal networks exhibit an anytime property: shallow subnetworks provide a quick answer, while deeper subnetworks, with higher latency, provide a more accurate answer.
    * Comments:
        * > (2016, DenseNet) FractalNets [17] repeatedly combine several parallel layer sequences with different number of convolutional blocks to obtain a large nominal depth, while maintaining many short paths in the network.
        * > (2016, DenseNet) FractalNets also achieve competitive results on several benchmark datasets using a wide network structure [17].
* [[DAG-CNN](https://arxiv.org/abs/1505.05232)]
    [[pdf](https://arxiv.org/pdf/1505.05232.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1505.05232/)]
    * Title: Multi-scale recognition with DAG-CNNs
    * Year: 20 May `2015`
    * Authors: Songfan Yang, Deva Ramanan
    * Institutions: [College of Electronics and Information Engineering, Sichuan University], [Deptment of Computer Science, University of California]
    * Abstract: We explore multi-scale convolutional neural nets (CNNs) for image classification. Contemporary approaches extract features from a single output layer. By extracting features from multiple layers, one can simultaneously reason about high, mid, and low-level features during classification. The resulting multi-scale architecture can itself be seen as a feed-forward model that is structured as a directed acyclic graph (DAG-CNNs). We use DAG-CNNs to learn a set of multiscale features that can be effectively shared between coarse and fine-grained classification tasks. While fine-tuning such models helps performance, we show that even "off-the-self" multiscale features perform quite well. We present extensive analysis and demonstrate state-of-the-art classification performance on three standard scene benchmarks (SUN397, MIT67, and Scene15). In terms of the heavily benchmarked MIT67 and Scene15 datasets, our results reduce the lowest previously-reported error by 23.9% and 9.5%, respectively.
    * Comments:
        * > (2016, DenseNet) In [9, 23, 30, 40], utilizing multi-level features in CNNs through skip-connnections has been found to be effective for various vision tasks.
* [[AdaNet](https://arxiv.org/abs/1607.01097)]
    [[pdf](https://arxiv.org/pdf/1607.01097.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1607.01097/)]
    * Title: AdaNet: Adaptive Structural Learning of Artificial Neural Networks
    * Year: 05 Jul `2016`
    * Authors: Corinna Cortes, Xavi Gonzalvo, Vitaly Kuznetsov, Mehryar Mohri, Scott Yang
    * Institutions: [Google Research, New York], [Courant Institute, New York]
    * Abstract: We present new algorithms for adaptively learning artificial neural networks. Our algorithms (AdaNet) adaptively learn both the structure of the network and its weights. They are based on a solid theoretical analysis, including data-dependent generalization guarantees that we prove and discuss in detail. We report the results of large-scale experiments with one of our algorithms on several binary classification tasks extracted from the CIFAR-10 dataset. The results demonstrate that our algorithm can automatically learn network structures with very competitive performance accuracies when compared with those achieved for neural networks found by standard approaches.
    * Comments:
        * > (2016, DenseNet) Parallel to our work, [1] derived a purely theoretical framework for networks with cross-layer connections similar to ours.
* [[Deeply-Supervised Networks (DSN)](https://arxiv.org/abs/1409.5185)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1409.5185.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1409.5185/)]
    * Title: Deeply-Supervised Nets
    * Year: 18 Sep `2014`
    * Authors: Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, Zhuowen Tu
    * Abstract: Our proposed deeply-supervised nets (DSN) method simultaneously minimizes classification error while making the learning process of hidden layers direct and transparent. We make an attempt to boost the classification performance by studying a new formulation in deep networks. Three aspects in convolutional neural networks (CNN) style architectures are being looked at: (1) transparency of the intermediate layers to the overall classification; (2) discriminativeness and robustness of learned features, especially in the early layers; (3) effectiveness in training due to the presence of the exploding and vanishing gradients. We introduce "companion objective" to the individual hidden layers, in addition to the overall objective at the output layer (a different strategy to layer-wise pre-training). We extend techniques from stochastic gradient methods to analyze our algorithm. The advantage of our method is evident and our experimental result on benchmark datasets shows significant performance gain over existing methods (e.g. all state-of-the-art results on MNIST, CIFAR-10, CIFAR-100, and SVHN).
    * Comments:
        * > (2015, ResNet) In [44, 24], a few intermediate layers are directly connected to auxiliary classifiers for addressing vanishing/exploding gradients.
        * > (2015, InceptionNetV3) Lee et al[11] argues that auxiliary classifiers promote more stable learning and better convergence.
        * > (2016, DenseNet) In Deeply Supervised Network (DSN) [20], internal layers are directly supervised by auxiliary classifiers, which can strengthen the gradients received by earlier layers.
* [[Deeply-Fused Networks (DFN)](https://arxiv.org/abs/1605.07716)]
    [[pdf](https://arxiv.org/pdf/1605.07716.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.07716/)]
    * Title: Deeply-Fused Nets
    * Year: 25 May `2016`
    * Authors: Jingdong Wang, Zhen Wei, Ting Zhang, Wenjun Zeng
    * Abstract: In this paper, we present a novel deep learning approach, deeply-fused nets. The central idea of our approach is deep fusion, i.e., combine the intermediate representations of base networks, where the fused output serves as the input of the remaining part of each base network, and perform such combinations deeply over several intermediate representations. The resulting deeply fused net enjoys several benefits. First, it is able to learn multi-scale representations as it enjoys the benefits of more base networks, which could form the same fused network, other than the initial group of base networks. Second, in our suggested fused net formed by one deep and one shallow base networks, the flows of the information from the earlier intermediate layer of the deep base network to the output and from the input to the later intermediate layer of the deep base network are both improved. Last, the deep and shallow base networks are jointly learnt and can benefit from each other. More interestingly, the essential depth of a fused net composed from a deep base network and a shallow base network is reduced because the fused net could be composed from a less deep base network, and thus training the fused net is less difficult than training the initial deep base network. Empirical results demonstrate that our approach achieves superior performance over two closely-related methods, ResNet and Highway, and competitive performance compared to the state-of-the-arts.
    * Comments:
        * > (2016, DenseNet) In [38], Deeply-Fused Nets (DFNs) were proposed to improve information flow by combining intermediate layers of different base networks.
* [[Augmenting Supervised Neural Networks with Unsupervised Objectives for Large-scale Image Classification](https://arxiv.org/abs/1606.06582)]
    [[pdf](https://arxiv.org/pdf/1606.06582.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1606.06582/)]
    * Title: Augmenting Supervised Neural Networks with Unsupervised Objectives for Large-scale Image Classification
    * Year: 21 Jun `2016`
    * Authors: Yuting Zhang, Kibok Lee, Honglak Lee
    * Abstract: Unsupervised learning and supervised learning are key research topics in deep learning. However, as high-capacity supervised neural networks trained with a large amount of labels have achieved remarkable success in many computer vision tasks, the availability of large-scale labeled images reduced the significance of unsupervised learning. Inspired by the recent trend toward revisiting the importance of unsupervised learning, we investigate joint supervised and unsupervised learning in a large-scale setting by augmenting existing neural networks with decoding pathways for reconstruction. First, we demonstrate that the intermediate activations of pretrained large-scale classification networks preserve almost all the information of input images except a portion of local spatial details. Then, by end-to-end training of the entire augmented architecture with the reconstructive objective, we show improvement of the network performance for supervised tasks. We evaluate several variants of autoencoders, including the recently proposed "what-where" autoencoder that uses the encoder pooling switches, to study the importance of the architecture design. Taking the 16-layer VGGNet trained under the ImageNet ILSVRC 2012 protocol as a strong baseline for image classification, our methods improve the validation-set accuracy by a noticeable margin.

### Ladder Networks

* [[Semi-Supervised Learning with Ladder Networks](https://arxiv.org/abs/1507.02672)]
    [[pdf](https://arxiv.org/pdf/1507.02672.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1507.02672/)]
    * Title: Semi-Supervised Learning with Ladder Networks
    * Year: 09 Jul `2015`
    * Authors: Antti Rasmus, Harri Valpola, Mikko Honkala, Mathias Berglund, Tapani Raiko
    * Abstract: We combine supervised learning with unsupervised learning in deep neural networks. The proposed model is trained to simultaneously minimize the sum of supervised and unsupervised cost functions by backpropagation, avoiding the need for layer-wise pre-training. Our work builds on the Ladder network proposed by Valpola (2015), which we extend by combining the model with supervision. We show that the resulting model reaches state-of-the-art performance in semi-supervised MNIST and CIFAR-10 classification, in addition to permutation-invariant MNIST classification with all labels.
* [[Deconstructing the Ladder Network Architecture](https://arxiv.org/abs/1511.06430)]
    [[pdf](https://arxiv.org/pdf/1511.06430.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06430/)]
    * Title: Deconstructing the Ladder Network Architecture
    * Year: 19 Nov `2015`
    * Authors: Mohammad Pezeshki, Linxi Fan, Philemon Brakel, Aaron Courville, Yoshua Bengio
    * Abstract: The Manual labeling of data is and will remain a costly endeavor. For this reason, semi-supervised learning remains a topic of practical importance. The recently proposed Ladder Network is one such approach that has proven to be very successful. In addition to the supervised objective, the Ladder Network also adds an unsupervised objective corresponding to the reconstruction costs of a stack of denoising autoencoders. Although the empirical results are impressive, the Ladder Network has many components intertwined, whose contributions are not obvious in such a complex architecture. In order to help elucidate and disentangle the different ingredients in the Ladder Network recipe, this paper presents an extensive experimental investigation of variants of the Ladder Network in which we replace or remove individual components to gain more insight into their relative importance. We find that all of the components are necessary for achieving optimal performance, but they do not contribute equally. For semi-supervised tasks, we conclude that the most important contribution is made by the lateral connection, followed by the application of noise, and finally the choice of what we refer to as the `combinator function' in the decoder path. We also find that as the number of labeled training examples increases, the lateral connections and reconstruction criterion become less important, with most of the improvement in generalization being due to the injection of noise in each layer. Furthermore, we present a new type of combinator function that outperforms the original design in both fully- and semi-supervised tasks, reducing record test error rates on Permutation-Invariant MNIST to 0.57% for the supervised setting, and to 0.97% and 1.0% for semi-supervised settings with 1000 and 100 labeled examples respectively.

### Residual Networks Family

> Deep Residual Network (ResNet) is one of the first works that successfully adopt skip connections, where each micro-block, a.k.a. residual function, is associated with a skip connection, called residual path. The residual path element-wisely adds the input features to the output of the same micro-block, making it a residual unit. Depending on the inner structure design of the micro-block, the residual network has developed into a family  of various architectures, including WRN, Inception-resnet, and ResNeXt. (2017, DPN)

* [[ResNet](https://arxiv.org/abs/1512.03385)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1512.03385.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1512.03385/)]
    * Title: Deep Residual Learning for Image Recognition
    * Year: 10 Dec `2015`
    * Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    * Institutions: [Microsoft Research]
    * Abstract: Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.
    * Comments:
        * > (2016, DenseNet) ResNets [11] and Highway Networks [33] bypass signal from one layer to the next via identity connections.
        * > (2016, InceptionNetV4) In [5], it is argued that residual connections are of inherent importance for training very deep architectures.
        * > (2016, PSPNet) Although theoretically the receptive field of ResNet [13] is already larger than the input image, it is shown by Zhou et al. [42] that the empirical receptive field of CNN is much smaller than the theoretical one especially on high-level layers.
        * > (2017, DPN) By revealing the equivalence of the state-of-the-art Residual Network (ResNet) and Densely Convolutional Network (DenseNet) within the HORNN framework, we find that ResNet enables feature re-usage while DenseNet enables new features exploration which are both important for learning good representations.
        * > (2017, DPN) Although the width of the densely connected path increases linearly as it goes deeper, causing the number of parameters to grow quadratically, DenseNet provides higher parameter efficiency compared with ResNet.
        * > (2017, ERFNet) It has been reported that non-bottleneck ResNets gain more accuracy from increased depth than the bottleneck versions, which indicates that they are not entirely equivalent and that the bottleneck design still suffers from the degradation problem.
        * > (2019, EfficientNetV1) Although several techniques, such as skip connections (He et al., 2016) and batch normalization (Ioffe & Szegedy, 2015), alleviate the training problem, the accuracy gain of very deep network diminishes: for example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more layers.
* [[ResNetV2](https://arxiv.org/abs/1603.05027)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1603.05027.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1603.05027/)]
    * Title: Identity Mappings in Deep Residual Networks
    * Year: 16 Mar `2016`
    * Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    * Abstract: Deep residual networks have emerged as a family of extremely deep architectures showing compelling accuracy and nice convergence behaviors. In this paper, we analyze the propagation formulations behind the residual building blocks, which suggest that the forward and backward signals can be directly propagated from one block to any other block, when using identity mappings as the skip connections and after-addition activation. A series of ablation experiments support the importance of these identity mappings. This motivates us to propose a new residual unit, which makes training easier and improves generalization. We report improved results using a 1001-layer ResNet on CIFAR-10 (4.62% error) and CIFAR-100, and a 200-layer ResNet on ImageNet. Code is available at: this https URL
* [[Resnet in Resnet](https://arxiv.org/abs/1603.08029)]
    [[pdf](https://arxiv.org/pdf/1603.08029.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1603.08029/)]
    * Title: Resnet in Resnet: Generalizing Residual Architectures
    * Year: 25 Mar `2016`
    * Authors: Sasha Targ, Diogo Almeida, Kevin Lyman
    * Abstract: Residual networks (ResNets) have recently achieved state-of-the-art on challenging computer vision tasks. We introduce Resnet in Resnet (RiR): a deep dual-stream architecture that generalizes ResNets and standard CNNs and is easily implemented with no computational overhead. RiR consistently improves performance over ResNets, outperforms architectures with similar amounts of augmentation on CIFAR-10, and establishes a new state-of-the-art on CIFAR-100.
* [[Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431)]
    [[pdf](https://arxiv.org/pdf/1605.06431.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.06431/)]
    * Title: Residual Networks Behave Like Ensembles of Relatively Shallow Networks
    * Year: 20 May `2016`
    * Authors: Andreas Veit, Michael Wilber, Serge Belongie
    * Abstract: In this work we propose a novel interpretation of residual networks showing that they can be seen as a collection of many paths of differing length. Moreover, residual networks seem to enable very deep networks by leveraging only the short paths during training. To support this observation, we rewrite residual networks as an explicit collection of paths. Unlike traditional models, paths through residual networks vary in length. Further, a lesion study reveals that these paths show ensemble-like behavior in the sense that they do not strongly depend on each other. Finally, and most surprising, most paths are shorter than one might expect, and only the short paths are needed during training, as longer paths do not contribute any gradient. For example, most of the gradient in a residual network with 110 layers comes from paths that are only 10-34 layers deep. Our results reveal one of the key characteristics that seem to enable the training of very deep networks: Residual networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of very deep networks.
* [[Wide Residual Networks (WRN)](https://arxiv.org/abs/1605.07146)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1605.07146.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.07146/)]
    * Title: Wide Residual Networks
    * Year: 23 May `2016`
    * Authors: Sergey Zagoruyko, Nikos Komodakis
    * Abstract: Deep residual networks were shown to be able to scale up to thousands of layers and still have improving performance. However, each fraction of a percent of improved accuracy costs nearly doubling the number of layers, and so training very deep residual networks has a problem of diminishing feature reuse, which makes these networks very slow to train. To tackle these problems, in this paper we conduct a detailed experimental study on the architecture of ResNet blocks, based on which we propose a novel architecture where we decrease depth and increase width of residual networks. We call the resulting network structures wide residual networks (WRNs) and show that these are far superior over their commonly used thin and very deep counterparts. For example, we demonstrate that even a simple 16-layer-deep wide residual network outperforms in accuracy and efficiency all previous deep residual networks, including thousand-layer-deep networks, achieving new state-of-the-art results on CIFAR, SVHN, COCO, and significant improvements on ImageNet. Our code and models are available at this https URL
    * Comments:
        * > (2016, DenseNet) In fact, simply increasing the number of filters in each layer of ResNets can improve its performance provided the depth is sufficient [41].
* [[ResNeXt](https://arxiv.org/abs/1611.05431)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1611.05431.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1611.05431/)]
    * Title: Aggregated Residual Transformations for Deep Neural Networks
    * Year: 16 Nov `2016`
    * Authors: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    * Institutions: [UC San Diego], [Facebook AI Research]
    * Abstract: We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call "cardinality" (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online.
    * Comments:
        * > (2017, ShuffleNetV1) The concept of group convolution, which was first introduced in AlexNet [21] for distributing the model over two GPUs, has been well demonstrated its effectiveness in ResNeXt [40].
        * > (2018, ESPNetV1) A ResNext module [14], shown in Fig. 3d, is a parallel version of the bottleneck module in ResNet [47] and is based on the principle of split-reduce-transform-expand-merge.
        * (2021, Swin Transformer V1) used depthwise convolutions.
* [[Collective Residual Unit (CRU)](https://arxiv.org/abs/1703.02180)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1703.02180.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.02180/)]
    * Title: Sharing Residual Units Through Collective Tensor Factorization in Deep Neural Networks
    * Year: 07 Mar `2017`
    * Authors: Chen Yunpeng, Jin Xiaojie, Kang Bingyi, Feng Jiashi, Yan Shuicheng
    * Abstract: Residual units are wildly used for alleviating optimization difficulties when building deep neural networks. However, the performance gain does not well compensate the model size increase, indicating low parameter efficiency in these residual units. In this work, we first revisit the residual function in several variations of residual units and demonstrate that these residual functions can actually be explained with a unified framework based on generalized block term decomposition. Then, based on the new explanation, we propose a new architecture, Collective Residual Unit (CRU), which enhances the parameter efficiency of deep neural networks through collective tensor factorization. CRU enables knowledge sharing across different residual units using shared factors. Experimental results show that our proposed CRU Network demonstrates outstanding parameter efficiency, achieving comparable classification performance to ResNet-200 with the model size of ResNet-50. By building a deeper network using CRU, we can achieve state-of-the-art single model classification accuracy on ImageNet-1k and Places365-Standard benchmark datasets. (Code and trained models are available on GitHub)
* [[Res2Net](https://arxiv.org/abs/1904.01169)]
    [[pdf](https://arxiv.org/pdf/1904.01169.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.01169/)]
    * Title: Res2Net: A New Multi-scale Backbone Architecture
    * Year: 02 Apr `2019`
    * Authors: Shang-Hua Gao, Ming-Ming Cheng, Kai Zhao, Xin-Yu Zhang, Ming-Hsuan Yang, Philip Torr
    * Abstract: Representing features at multiple scales is of great importance for numerous vision tasks. Recent advances in backbone convolutional neural networks (CNNs) continually demonstrate stronger multi-scale representation ability, leading to consistent performance gains on a wide range of applications. However, most existing methods represent the multi-scale features in a layer-wise manner. In this paper, we propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models on widely-used datasets, e.g., CIFAR-100 and ImageNet. Further ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, further verify the superiority of the Res2Net over the state-of-the-art baseline methods. The source code and trained models are available on this https URL.
* [[ResNeSt](https://arxiv.org/abs/2004.08955)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2004.08955.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2004.08955/)]
    * Title: ResNeSt: Split-Attention Networks
    * Year: 19 Apr `2020`
    * Authors: Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Haibin Lin, Zhi Zhang, Yue Sun, Tong He, Jonas Mueller, R. Manmatha, Mu Li, Alexander Smola
    * Abstract: It is well known that featuremap attention and multi-path representation are important for visual recognition. In this paper, we present a modularized architecture, which applies the channel-wise attention on different network branches to leverage their success in capturing cross-feature interactions and learning diverse representations. Our design results in a simple and unified computation block, which can be parameterized using only a few variables. Our model, named ResNeSt, outperforms EfficientNet in accuracy and latency trade-off on image classification. In addition, ResNeSt has achieved superior transfer learning results on several public benchmarks serving as the backbone, and has been adopted by the winning entries of COCO-LVIS challenge. The source code for complete system and pretrained models are publicly available.
    * Comments:
        * > RegNet (Radosavovic et al., 2020), ResNeSt (Zhang et al., 2020), TResNet (Ridnik et al., 2020), and EfficientNet-X (Li et al., 2021) focus on GPU and/or TPU inference speed. (EfficientNetV2, 2021)
* [[ResNet-RS](https://arxiv.org/abs/2103.07579)]
    [[pdf](https://arxiv.org/pdf/2103.07579.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.07579/)]
    * Title: Revisiting ResNets: Improved Training and Scaling Strategies
    * Year: 13 Mar `2021`
    * Authors: Irwan Bello, William Fedus, Xianzhi Du, Ekin D. Cubuk, Aravind Srinivas, Tsung-Yi Lin, Jonathon Shlens, Barret Zoph
    * Abstract: Novel computer vision architectures monopolize the spotlight, but the impact of the model architecture is often conflated with simultaneous changes to training methodology and scaling strategies. Our work revisits the canonical ResNet (He et al., 2015) and studies these three aspects in an effort to disentangle them. Perhaps surprisingly, we find that training and scaling strategies may matter more than architectural changes, and further, that the resulting ResNets match recent state-of-the-art models. We show that the best performing scaling strategy depends on the training regime and offer two new scaling strategies: (1) scale model depth in regimes where overfitting can occur (width scaling is preferable otherwise); (2) increase image resolution more slowly than previously recommended (Tan & Le, 2019). Using improved training and scaling strategies, we design a family of ResNet architectures, ResNet-RS, which are 1.7x - 2.7x faster than EfficientNets on TPUs, while achieving similar accuracies on ImageNet. In a large-scale semi-supervised learning setup, ResNet-RS achieves 86.2% top-1 ImageNet accuracy, while being 4.7x faster than EfficientNet NoisyStudent. The training techniques improve transfer performance on a suite of downstream tasks (rivaling state-of-the-art self-supervised algorithms) and extend to video classification on Kinetics-400. We recommend practitioners use these simple revised ResNets as baselines for future research.
    * Comments:
        * > ResNet-RS (Bello et al., 2021) improves training efficiency by optimizing scaling hyperparameters. (EfficientNetV2, 2021)
        * > Lambda Networks (Bello, 2021), NFNets (Brock et al., 2021), BoTNets (Srinivas et al., 2021), ResNet-RS (Bello et al., 2021) focus on TPU training speed. (EfficientNetV2, 2021)
* [[NFNet](https://arxiv.org/abs/2102.06171)]
    [[pdf](https://arxiv.org/pdf/2102.06171.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2102.06171/)]
    * Title: High-Performance Large-Scale Image Recognition Without Normalization
    * Year: 11 Feb `2021`
    * Authors: Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan
    * Abstract: Batch normalization is a key component of most image classification models, but it has many undesirable properties stemming from its dependence on the batch size and interactions between examples. Although recent work has succeeded in training deep ResNets without normalization layers, these models do not match the test accuracies of the best batch-normalized networks, and are often unstable for large learning rates or strong data augmentations. In this work, we develop an adaptive gradient clipping technique which overcomes these instabilities, and design a significantly improved class of Normalizer-Free ResNets. Our smaller models match the test accuracy of an EfficientNet-B7 on ImageNet while being up to 8.7x faster to train, and our largest models attain a new state-of-the-art top-1 accuracy of 86.5%. In addition, Normalizer-Free models attain significantly better performance than their batch-normalized counterparts when finetuning on ImageNet after large-scale pre-training on a dataset of 300 million labeled images, with our best models obtaining an accuracy of 89.2%. Our code is available at this https URL deepmind-research/tree/master/nfnets
    * Comments:
        * > (2021, EfficientNetV2) NFNets (Brock et al., 2021) aim to improve training efficiency by removing the expensive batch normalization.
        * > (2021, EfficientNetV2) Lambda Networks (Bello, 2021), NFNets (Brock et al., 2021), BoTNets (Srinivas et al., 2021), ResNet-RS (Bello et al., 2021) focus on TPU training speed.
* [[TResNet](https://arxiv.org/abs/2003.13630)]
    [[pdf](https://arxiv.org/pdf/2003.13630.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2003.13630/)]
    * Title: TResNet: High Performance GPU-Dedicated Architecture
    * Year: 30 Mar `2020`
    * Authors: Tal Ridnik, Hussam Lawen, Asaf Noy, Emanuel Ben Baruch, Gilad Sharir, Itamar Friedman
    * Abstract: Many deep learning models, developed in recent years, reach higher ImageNet accuracy than ResNet50, with fewer or comparable FLOPS count. While FLOPs are often seen as a proxy for network efficiency, when measuring actual GPU training and inference throughput, vanilla ResNet50 is usually significantly faster than its recent competitors, offering better throughput-accuracy trade-off. In this work, we introduce a series of architecture modifications that aim to boost neural networks' accuracy, while retaining their GPU training and inference efficiency. We first demonstrate and discuss the bottlenecks induced by FLOPs-optimizations. We then suggest alternative designs that better utilize GPU structure and assets. Finally, we introduce a new family of GPU-dedicated models, called TResNet, which achieve better accuracy and efficiency than previous ConvNets. Using a TResNet model, with similar GPU throughput to ResNet50, we reach 80.8 top-1 accuracy on ImageNet. Our TResNet models also transfer well and achieve state-of-the-art accuracy on competitive single-label classification datasets such as Stanford cars (96.0%), CIFAR-10 (99.0%), CIFAR-100 (91.5%) and Oxford-Flowers (99.1%). They also perform well on multi-label classification and object detection tasks. Implementation is available at: this https URL.
    * Comments:
        * > RegNet (Radosavovic et al., 2020), ResNeSt (Zhang et al., 2020), TResNet (Ridnik et al., 2020), and EfficientNet-X (Li et al., 2021) focus on GPU and/or TPU inference speed. (EfficientNetV2, 2021)

### Dense Topology Framework

* [[DenseNet](https://arxiv.org/abs/1608.06993)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1608.06993.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1608.06993/)]
    * Title: Densely Connected Convolutional Networks
    * Year: 25 Aug `2016`
    * Authors: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    * Institutions: [Cornell University], [Tsinghua University], [Facebook AI Research]
    * Abstract: Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance. Code and pre-trained models are available at this https URL .
    * Comments:
        * > (2017, DPN) By revealing the equivalence of the state-of-the-art Residual Network (ResNet) and Densely Convolutional Network (DenseNet) within the HORNN framework, we find that ResNet enables feature re-usage while DenseNet enables new features exploration which are both important for learning good representations.
        * > (2017, DPN) Although the width of the densely connected path increases linearly as it goes deeper, causing the number of parameters to grow quadratically, DenseNet provides higher parameter efficiency compared with ResNet.
        * > (2021, EfficientNetV2) Many works, such as DenseNet (Huang et al., 2017) and EfficientNet (Tan & Le, 2019a), focus on parameter efficiency, aiming to achieve better accuracy with less parameters.
* [[Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex](https://arxiv.org/abs/1604.03640)]
    [[pdf](https://arxiv.org/pdf/1604.03640.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1604.03640/)]
    * Title: Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex
    * Year: 13 Apr `2016`
    * Authors: Qianli Liao, Tomaso Poggio
    * Abstract: We discuss relations between Residual Networks (ResNet), Recurrent Neural Networks (RNNs) and the primate visual cortex. We begin with the observation that a special type of shallow RNN is exactly equivalent to a very deep ResNet with weight sharing among the layers. A direct implementation of such a RNN, although having orders of magnitude fewer parameters, leads to a performance similar to the corresponding ResNet. We propose 1) a generalization of both RNN and ResNet architectures and 2) the conjecture that a class of moderately deep RNNs is a biologically-plausible model of the ventral stream in visual cortex. We demonstrate the effectiveness of the architectures by testing them on the CIFAR-10 and ImageNet dataset.
* [[HORNN](https://arxiv.org/abs/1605.00064)]
    [[pdf](https://arxiv.org/pdf/1605.00064.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.00064/)]
    * Title: Higher Order Recurrent Neural Networks
    * Year: 30 Apr `2016`
    * Authors: Rohollah Soltani, Hui Jiang
    * Abstract: In this paper, we study novel neural network structures to better model long term dependency in sequential data. We propose to use more memory units to keep track of more preceding states in recurrent neural networks (RNNs), which are all recurrently fed to the hidden layers as feedback through different weighted paths. By extending the popular recurrent structure in RNNs, we provide the models with better short-term memory mechanism to learn long term dependency in sequences. Analogous to digital filters in signal processing, we call these structures as higher order RNNs (HORNNs). Similar to RNNs, HORNNs can also be learned using the back-propagation through time method. HORNNs are generally applicable to a variety of sequence modelling tasks. In this work, we have examined HORNNs for the language modeling task using two popular data sets, namely the Penn Treebank (PTB) and English text8 data sets. Experimental results have shown that the proposed HORNNs yield the state-of-the-art performance on both data sets, significantly outperforming the regular RNNs as well as the popular LSTMs.
* [[Dual Path Networks (DPN)](https://arxiv.org/abs/1707.01629)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1707.01629.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1707.01629/)]
    * Title: Dual Path Networks
    * Year: 06 Jul `2017`
    * Authors: Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng
    * Abstract: In this work, we present a simple, highly efficient and modularized Dual Path Network (DPN) for image classification which presents a new topology of connection paths internally. By revealing the equivalence of the state-of-the-art Residual Network (ResNet) and Densely Convolutional Network (DenseNet) within the HORNN framework, we find that ResNet enables feature re-usage while DenseNet enables new features exploration which are both important for learning good representations. To enjoy the benefits from both path topologies, our proposed Dual Path Network shares common features while maintaining the flexibility to explore new features through dual path architectures. Extensive experiments on three benchmark datasets, ImagNet-1k, Places365 and PASCAL VOC, clearly demonstrate superior performance of the proposed DPN over state-of-the-arts. In particular, on the ImagNet-1k dataset, a shallow DPN surpasses the best ResNeXt-101(64x4d) with 26% smaller model size, 25% less computational cost and 8% lower memory consumption, and a deeper DPN (DPN-131) further pushes the state-of-the-art single model performance with about 2 times faster training speed. Experiments on the Places365 large-scale scene dataset, PASCAL VOC detection dataset, and PASCAL VOC segmentation dataset also demonstrate its consistently better performance than DenseNet, ResNet and the latest ResNeXt model over various applications.
* [[Mixed Link Networks](https://arxiv.org/abs/1802.01808)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1802.01808.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1802.01808/)]
    * Title: Mixed Link Networks
    * Year: 06 Feb `2018`
    * Authors: Wenhai Wang, Xiang Li, Jian Yang, Tong Lu
    * Abstract: Basing on the analysis by revealing the equivalence of modern networks, we find that both ResNet and DenseNet are essentially derived from the same "dense topology", yet they only differ in the form of connection -- addition (dubbed "inner link") vs. concatenation (dubbed "outer link"). However, both two forms of connections have the superiority and insufficiency. To combine their advantages and avoid certain limitations on representation learning, we present a highly efficient and modularized Mixed Link Network (MixNet) which is equipped with flexible inner link and outer link modules. Consequently, ResNet, DenseNet and Dual Path Network (DPN) can be regarded as a special case of MixNet, respectively. Furthermore, we demonstrate that MixNets can achieve superior efficiency in parameter over the state-of-the-art architectures on many competitive datasets like CIFAR-10/100, SVHN and ImageNet.

### Grouped Convolutions

> Grouped convolutions have proven to be a popular approach for increasing the cardinality of learned transformations [Deep Roots], [ResNeXt]. (SENet, 2017)

* [[Deep Roots](https://arxiv.org/abs/1605.06489)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1605.06489.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.06489/)]
    * Title: Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups
    * Year: 20 May `2016`
    * Authors: Yani Ioannou, Duncan Robertson, Roberto Cipolla, Antonio Criminisi
    * Abstract: We propose a new method for creating computationally efficient and compact convolutional neural networks (CNNs) using a novel sparse connection structure that resembles a tree root. This allows a significant reduction in computational cost and number of parameters compared to state-of-the-art deep CNNs, without compromising accuracy, by exploiting the sparsity of inter-layer filter dependencies. We validate our approach by using it to train more efficient variants of state-of-the-art CNN architectures, evaluated on the CIFAR10 and ILSVRC datasets. Our results show similar or higher accuracy than the baseline architectures with much less computation, as measured by CPU and GPU timings. For example, for ResNet 50, our model has 40% fewer parameters, 45% fewer floating point operations, and is 31% (12%) faster on a CPU (GPU). For the deeper ResNet 200 our model has 25% fewer floating point operations and 44% fewer parameters, while maintaining state-of-the-art accuracy. For GoogLeNet, our model has 7% fewer parameters and is 21% (16%) faster on a CPU (GPU).
* Aggregated Residual Transformations for Deep Neural Networks (ResNeXt)
* Inception Networks.

### InceptionNet Series

Overview:

> The Inception models have evolved over time, but an important common property is a *split-transform-merge* strategy. In an Inception module, the input is split into a few lower-dimensional embeddings (by $1 \times 1$ convolutions), transformed by a set of specialized filters ($3 \times 3$, $5 \times 5$, etc.), and merged by concatenation. It can be shown that the solution space of this architecture is a strict subspace of the solution space of a single large layer (e.g., $5 \times 5$) operating on a high-dimensional embedding. The split-transform-merge behavior of Inception modules is expected to approach the representational power of large and dense layers, but at a considerably lower computational complexity. (ResNeXt, 2016)

> The Inception models are successful multi-branch architectures where each branch is carefully customized. ResNets can be thought of as two-branch networks where one branch is the identity mapping. Deep neural decision forests are tree-patterned multi-branch networks with learned splitting functions. (ResNeXt, 2016)

* Comments from the Xception Network
    * > (2016, Xception) Inception itself was inspired by the earlier Network-In-Network architecture [11].
    * > (2016, Xception) The fundamental building block of Inception-style models is the Inception module, of which several different versions exist. In figure 1 we show the canonical form of an Inception module, as found in the Inception V3 architecture. An Inception model can be understood as a stack of such modules. This is a departure from earlier VGG-style networks which were stacks of simple convolution layers.
    * > (2016, Xception) In effect, the fundamental hypothesis behind Inception is that cross-channel correlations and spatial correlations are sufficiently decoupled that it is preferable not to map them jointly.

> More flexible compositions of operators can be achieved with multi-branch convolutions [5], [6], [20], [21], which can be viewed as a natural extension of the grouping operator. (SENet, 2017)

> Inception modules [11–13] are built on the principle of split-reduce-transform-merge. These modules are usually heterogeneous in number of channels and kernel size (e.g. some of the modules are composed of standard and factored convolutions). (ESPNetV1, 2018)

* [[InceptionNetV1/GoogLeNet](https://arxiv.org/abs/1409.4842)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1409.4842.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1409.4842/)]
    * Title: Going Deeper with Convolutions
    * Year: 17 Sep `2014`
    * Authors: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
    * Institutions: [Google Inc.], [University of North Carolina, Chapel Hill], [University of Michigan]
    * Abstract: We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.
    * Comments:
        * > (2014, VGGNet) The spatial resolution of the feature maps is reduced more aggressively in the first layers to decrease the amount of computation.
        * > (2015, ResNet) In [44, 24], a few intermediate layers are directly connected to auxiliary classifiers for addressing vanishing/exploding gradients.
        * > (2015, ResNet) In [44], an "inception" layer is composed of a shortcut branch and a few deeper branches.
        * > (2015, InceptionNetV3) Much of the original gains of the GoogLeNet network [20] arise from a very generous use of dimension reduction. This can be viewed as a special case of factorizing convolutions in a computationally efficient manner.
        * > (2015, InceptionNetV3) [20] has introduced the notion of auxiliary classifiers to improve the convergence of very deep networks. The original motivation was to push useful gradients to the lower layers to make them immediately useful and improve the convergence during training by combating the vanishing gradient problem in very deep networks.
        * > (2016, InceptionNetV4) Later the architecture was improved by additional factorization ideas in the third iteration [15] which will be referred to as Inception-v3 in this report.
* [[InceptionNetV2/Batch Normalization](https://arxiv.org/abs/1502.03167)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1502.03167.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1502.03167/)]
    * Title: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    * Year: 11 Feb `2015`
    * Authors: Sergey Ioffe, Christian Szegedy
    * Institutions: [Google Inc.]
    * Abstract: Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.
    * Comments:
        * > (2017, MobileNetV1) MobileNets are built primarily from depthwise separable convolutions initially introduced in [26] and subsequently used in Inception models [13] to reduce the computation in the first few layers.
* [[InceptionNetV3](https://arxiv.org/abs/1512.00567)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1512.00567.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1512.00567/)]
    * Title: Rethinking the Inception Architecture for Computer Vision
    * Year: 02 Dec `2015`
    * Authors: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
    * Institutions: [Google Inc.], [University College London]
    * Abstract: Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set.
* [[InceptionNetV4/Inception-ResNet](https://arxiv.org/abs/1602.07261)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1602.07261.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1602.07261/)]
    * Title: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    * Year: 23 Feb `2016`
    * Authors: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
    * Institutions: [Google Inc.]
    * Abstract: Very deep convolutional networks have been central to the largest advances in image recognition performance in recent years. One example is the Inception architecture that has been shown to achieve very good performance at relatively low computational cost. Recently, the introduction of residual connections in conjunction with a more traditional architecture has yielded state-of-the-art performance in the 2015 ILSVRC challenge; its performance was similar to the latest generation Inception-v3 network. This raises the question of whether there are any benefit in combining the Inception architecture with residual connections. Here we give clear empirical evidence that training with residual connections accelerates the training of Inception networks significantly. There is also some evidence of residual Inception networks outperforming similarly expensive Inception networks without residual connections by a thin margin. We also present several new streamlined architectures for both residual and non-residual Inception networks. These variations improve the single-frame recognition performance on the ILSVRC 2012 classification task significantly. We further demonstrate how proper activation scaling stabilizes the training of very wide residual Inception networks. With an ensemble of three residual and one Inception-v4, we achieve 3.08 percent top-5 error on the test set of the ImageNet classification (CLS) challenge
* [[PolyInception](https://arxiv.org/abs/1611.05725)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1611.05725.pdf)]
    [vanity]
    * Title: PolyNet: A Pursuit of Structural Diversity in Very Deep Networks
    * Year: 17 Nov `2016`
    * Authors: Xingcheng Zhang, Zhizhong Li, Chen Change Loy, Dahua Lin
    * Abstract: A number of studies have shown that increasing the depth or width of convolutional networks is a rewarding approach to improve the performance of image recognition. In our study, however, we observed difficulties along both directions. On one hand, the pursuit for very deep networks is met with a diminishing return and increased training difficulty; on the other hand, widening a network would result in a quadratic growth in both computational cost and memory demand. These difficulties motivate us to explore structural diversity in designing deep networks, a new dimension beyond just depth and width. Specifically, we present a new family of modules, namely the PolyInception, which can be flexibly inserted in isolation or in a composition as replacements of different parts of a network. Choosing PolyInception modules with the guidance of architectural efficiency can improve the expressive power while preserving comparable computational cost. The Very Deep PolyNet, designed following this direction, demonstrates substantial improvements over the state-of-the-art on the ILSVRC 2012 benchmark. Compared to Inception-ResNet-v2, it reduces the top-5 validation error on single crops from 4.9% to 4.25%, and that on multi-crops from 3.7% to 3.45%.

### Multi-Column Networks

* [[Multi-Column Networks](https://arxiv.org/abs/1202.2745)]
    [[pdf](https://arxiv.org/pdf/1202.2745.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1202.2745/)]
    * Title: Multi-column Deep Neural Networks for Image Classification
    * Year: 13 Feb `2012`
    * Authors: Dan Cireşan, Ueli Meier, Juergen Schmidhuber
    * Abstract: Traditional methods of computer vision and machine learning cannot match human performance on tasks such as the recognition of handwritten digits or traffic signs. Our biologically plausible deep artificial neural network architectures can. Small (often minimal) receptive fields of convolutional winner-take-all neurons yield large network depth, resulting in roughly as many sparsely connected neural layers as found in mammals between retina and visual cortex. Only winner neurons are trained. Several deep neural columns become experts on inputs preprocessed in different ways; their predictions are averaged. Graphics cards allow for fast training. On the very competitive MNIST handwriting benchmark, our method is the first to achieve near-human performance. On a traffic sign recognition benchmark it outperforms humans by a factor of two. We also improve the state-of-the-art on a plethora of common image classification benchmarks.
    * Comments:
        * > Within just a few years, the top-5 image classification accuracy on the 1000-class ImageNet dataset has increased from ~84% [1] to ~95% [2, 3] using deeper networks with rather small receptive fields [4, 5]. (Training Very Deep Networks, 2015)
* [[High-Performance Neural Networks for Visual Object Classification](https://arxiv.org/abs/1102.0183)]
    [[pdf](https://arxiv.org/pdf/1102.0183.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1102.0183/)]
    * Title: High-Performance Neural Networks for Visual Object Classification
    * Year: 01 Feb `2011`
    * Authors: Dan C. Cireşan, Ueli Meier, Jonathan Masci, Luca M. Gambardella, Jürgen Schmidhuber
    * Abstract: We present a fast, fully parameterizable GPU implementation of Convolutional Neural Network variants. Our feature extractors are neither carefully designed nor pre-wired, but rather learned in a supervised way. Our deep hierarchical architectures achieve the best published results on benchmarks for object classification (NORB, CIFAR10) and handwritten digit recognition (MNIST), with error rates of 2.53%, 19.51%, 0.35%, respectively. Deep nets trained by simple back-propagation perform better than more shallow ones. Learning is surprisingly rapid. NORB is completely trained within five epochs. Test error rates on MNIST drop to 2.42%, 0.97% and 0.48% after 1, 3 and 17 epochs, respectively.
* [[Deep Columnar Convolutional Neural Network](https://www.researchgate.net/publication/305361867_Deep_Columnar_Convolutional_Neural_Network)]
    [[pdf](https://www.researchgate.net/profile/Somshubra-Majumdar/publication/305361867_Deep_Columnar_Convolutional_Neural_Network/links/5795972508aec89db7b462d5/Deep-Columnar-Convolutional-Neural-Network.pdf)]
    * Title: Deep Columnar Convolutional Neural Network

## Research on Convolutional Kernels

### Deformable Kernels

* [[DeformableNetV1](https://arxiv.org/abs/1703.06211)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1703.06211.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.06211/)]
    * Title: Deformable Convolutional Networks
    * Year: 17 Mar `2017`
    * Authors: Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei
    * Abstract: Convolutional neural networks (CNNs) are inherently limited to model geometric transformations due to the fixed geometric structures in its building modules. In this work, we introduce two new modules to enhance the transformation modeling capacity of CNNs, namely, deformable convolution and deformable RoI pooling. Both are based on the idea of augmenting the spatial sampling locations in the modules with additional offsets and learning the offsets from target tasks, without additional supervision. The new modules can readily replace their plain counterparts in existing CNNs and can be easily trained end-to-end by standard back-propagation, giving rise to deformable convolutional networks. Extensive experiments validate the effectiveness of our approach on sophisticated vision tasks of object detection and semantic segmentation. The code would be released.
* [[DeformableNetV2](https://arxiv.org/abs/1811.11168)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1811.11168.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1811.11168/)]
    * Title: Deformable ConvNets v2: More Deformable, Better Results
    * Year: 27 Nov `2018`
    * Authors: Xizhou Zhu, Han Hu, Stephen Lin, Jifeng Dai
    * Abstract: The superior performance of Deformable Convolutional Networks arises from its ability to adapt to the geometric variations of objects. Through an examination of its adaptive behavior, we observe that while the spatial support for its neural features conforms more closely than regular ConvNets to object structure, this support may nevertheless extend well beyond the region of interest, causing features to be influenced by irrelevant image content. To address this problem, we present a reformulation of Deformable ConvNets that improves its ability to focus on pertinent image regions, through increased modeling power and stronger training. The modeling power is enhanced through a more comprehensive integration of deformable convolution within the network, and by introducing a modulation mechanism that expands the scope of deformation modeling. To effectively harness this enriched modeling capability, we guide network training via a proposed feature mimicking scheme that helps the network to learn features that reflect the object focus and classification power of R-CNN features. With the proposed contributions, this new version of Deformable ConvNets yields significant performance gains over the original model and produces leading results on the COCO benchmark for object detection and instance segmentation.

### Dilated Kernels

* [[Dilated Convolutions](https://arxiv.org/abs/1511.07122)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1511.07122.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.07122/)]
    * Title: Multi-Scale Context Aggregation by Dilated Convolutions
    * Year: 23 Nov `2015`
    * Authors: Fisher Yu, Vladlen Koltun
    * Institution: [Princeton University], [Intel Labs]
    * Abstract: State-of-the-art models for semantic segmentation are based on adaptations of convolutional networks that had originally been designed for image classification. However, dense prediction and image classification are structurally different. In this work, we develop a new convolutional network module that is specifically designed for dense prediction. The presented module uses dilated convolutions to systematically aggregate multi-scale contextual information without losing resolution. The architecture is based on the fact that dilated convolutions support exponential expansion of the receptive field without loss of resolution or coverage. We show that the presented context module increases the accuracy of state-of-the-art semantic segmentation systems. In addition, we examine the adaptation of image classification networks to dense prediction and show that simplifying the adapted network can increase accuracy.
    * Comments:
        * > Yu and Koltun [18] stacked dilated convolution layers with increasing dilation rate to learn contextual representations from a large effective receptive field. (ESPNetV1, 2018)
* [[ESPNetV1](https://arxiv.org/abs/1803.06815)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1803.06815.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1803.06815/)]
    * Title: ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
    * Year: 19 Mar `2018`
    * Authors: Sachin Mehta, Mohammad Rastegari, Anat Caspi, Linda Shapiro, Hannaneh Hajishirzi
    * Abstract: We introduce a fast and efficient convolutional neural network, ESPNet, for semantic segmentation of high resolution images under resource constraints. ESPNet is based on a new convolutional module, efficient spatial pyramid (ESP), which is efficient in terms of computation, memory, and power. ESPNet is 22 times faster (on a standard GPU) and 180 times smaller than the state-of-the-art semantic segmentation network PSPNet, while its category-wise accuracy is only 8% less. We evaluated ESPNet on a variety of semantic segmentation datasets including Cityscapes, PASCAL VOC, and a breast biopsy whole slide image dataset. Under the same constraints on memory and computation, ESPNet outperforms all the current efficient CNN networks such as MobileNet, ShuffleNet, and ENet on both standard metrics and our newly introduced performance metrics that measure efficiency on edge devices. Our network can process high resolution images at a rate of 112 and 9 frames per second on a standard GPU and edge device, respectively.
* [[ESPNetv2](https://arxiv.org/abs/1811.11431)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1811.11431.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1811.11431/)]
    * Title: ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network
    * Year: 28 Nov `2018`
    * Authors: Sachin Mehta, Mohammad Rastegari, Linda Shapiro, Hannaneh Hajishirzi
    * Abstract: We introduce a light-weight, power efficient, and general purpose convolutional neural network, ESPNetv2, for modeling visual and sequential data. Our network uses group point-wise and depth-wise dilated separable convolutions to learn representations from a large effective receptive field with fewer FLOPs and parameters. The performance of our network is evaluated on four different tasks: (1) object classification, (2) semantic segmentation, (3) object detection, and (4) language modeling. Experiments on these tasks, including image classification on the ImageNet and language modeling on the PenTree bank dataset, demonstrate the superior performance of our method over the state-of-the-art methods. Our network outperforms ESPNet by 4-5% and has 2-4x fewer FLOPs on the PASCAL VOC and the Cityscapes dataset. Compared to YOLOv2 on the MS-COCO object detection, ESPNetv2 delivers 4.4% higher accuracy with 6x fewer FLOPs. Our experiments show that ESPNetv2 is much more power efficient than existing state-of-the-art efficient methods including ShuffleNets and MobileNets. Our code is open-source and available at this https URL
* [[Dilated Residual Networks](https://arxiv.org/abs/1705.09914)]
    [[pdf](https://arxiv.org/pdf/1705.09914.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1705.09914/)]
    * Title: Dilated Residual Networks
    * Year: 28 May `2017`
    * Authors: Fisher Yu, Vladlen Koltun, Thomas Funkhouser
    * Abstract: Convolutional networks for image classification progressively reduce resolution until the image is represented by tiny feature maps in which the spatial structure of the scene is no longer discernible. Such loss of spatial acuity can limit image classification accuracy and complicate the transfer of the model to downstream applications that require detailed scene understanding. These problems can be alleviated by dilation, which increases the resolution of output feature maps without reducing the receptive field of individual neurons. We show that dilated residual networks (DRNs) outperform their non-dilated counterparts in image classification without increasing the model's depth or complexity. We then study gridding artifacts introduced by dilation, develop an approach to removing these artifacts (`degridding'), and show that this further increases the performance of DRNs. In addition, we show that the accuracy advantage of DRNs is further magnified in downstream applications such as object localization and semantic segmentation.

### Adaptive Kernels

* [[Dynamic Filter Networks](https://arxiv.org/abs/1605.09673)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1605.09673.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.09673/)]
    * Title: Dynamic Filter Networks
    * Year: 31 May `2016`
    * Authors: Bert De Brabandere, Xu Jia, Tinne Tuytelaars, Luc Van Gool
    * Abstract: In a traditional convolutional layer, the learned filters stay fixed after training. In contrast, we introduce a new framework, the Dynamic Filter Network, where filters are generated dynamically conditioned on an input. We show that this architecture is a powerful one, with increased flexibility thanks to its adaptive nature, yet without an excessive increase in the number of model parameters. A wide variety of filtering operations can be learned this way, including local spatial transformations, but also others like selective (de)blurring or adaptive feature extraction. Moreover, multiple such layers can be combined, e.g. in a recurrent architecture. We demonstrate the effectiveness of the dynamic filter network on the tasks of video and stereo prediction, and reach state-of-the-art performance on the moving MNIST dataset with a much smaller model. By visualizing the learned filters, we illustrate that the network has picked up flow information by only looking at unlabelled training data. This suggests that the network can be used to pretrain networks for various supervised tasks in an unsupervised way, like optical flow and depth estimation.
* [[SKNet](https://arxiv.org/abs/1903.06586)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1903.06586.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1903.06586/)]
    * Title: Selective Kernel Networks
    * Year: 15 Mar `2019`
    * Authors: Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang
    * Abstract: In standard Convolutional Neural Networks (CNNs), the receptive fields of artificial neurons in each layer are designed to share the same size. It is well-known in the neuroscience community that the receptive field size of visual cortical neurons are modulated by the stimulus, which has been rarely considered in constructing CNNs. We propose a dynamic selection mechanism in CNNs that allows each neuron to adaptively adjust its receptive field size based on multiple scales of input information. A building block called Selective Kernel (SK) unit is designed, in which multiple branches with different kernel sizes are fused using softmax attention that is guided by the information in these branches. Different attentions on these branches yield different sizes of the effective receptive fields of neurons in the fusion layer. Multiple SK units are stacked to a deep network termed Selective Kernel Networks (SKNets). On the ImageNet and CIFAR benchmarks, we empirically show that SKNet outperforms the existing state-of-the-art architectures with lower model complexity. Detailed analyses show that the neurons in SKNet can capture target objects with different scales, which verifies the capability of neurons for adaptively adjusting their receptive field sizes according to the input. The code and models are available at this https URL.

### Cross-Channel Correlations

> In prior work, cross channel correlations are typically mapped as new combinations of features, either independently of spatial structure [speeding up], [Xception] or jointly by using standard convolutional filters [Network in Network] with $1 \times 1$ convolutions. (SENet, 2017)

* Speeding up Convolutional Neural Networks with Low Rank Expansions
* Xception: Deep Learning with Depthwise Separable Convolutions
* NIN, see `mlp.md`.

## Research on Attention Mechanism

### Survey

* [[Attention Mechanisms in Computer Vision: A Survey](https://arxiv.org/abs/2111.07624)]
    [[pdf](https://arxiv.org/pdf/2111.07624.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2111.07624/)]
    * Title: Attention Mechanisms in Computer Vision: A Survey
    * Year: 15 Nov `2021`
    * Authors: Meng-Hao Guo, Tian-Xing Xu, Jiang-Jiang Liu, Zheng-Ning Liu, Peng-Tao Jiang, Tai-Jiang Mu, Song-Hai Zhang, Ralph R. Martin, Ming-Ming Cheng, Shi-Min Hu
    * Abstract: Humans can naturally and effectively find salient regions in complex scenes. Motivated by this observation, attention mechanisms were introduced into computer vision with the aim of imitating this aspect of the human visual system. Such an attention mechanism can be regarded as a dynamic weight adjustment process based on features of the input image. Attention mechanisms have achieved great success in many visual tasks, including image classification, object detection, semantic segmentation, video understanding, image generation, 3D vision, multi-modal tasks and self-supervised learning. In this survey, we provide a comprehensive review of various attention mechanisms in computer vision and categorize them according to approach, such as channel attention, spatial attention, temporal attention and branch attention; a related repository this https URL is dedicated to collecting related work. We also suggest future directions for attention mechanism research.

### Unknown

* [[Gather-Excite](https://arxiv.org/abs/1810.12348)]
    [[pdf](https://arxiv.org/pdf/1810.12348.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1810.12348/)]
    * Title: Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks
    * Year: 29 Oct `2018`
    * Authors: Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Andrea Vedaldi
    * Abstract: While the use of bottom-up local operators in convolutional neural networks (CNNs) matches well some of the statistics of natural images, it may also prevent such models from capturing contextual long-range feature interactions. In this work, we propose a simple, lightweight approach for better context exploitation in CNNs. We do so by introducing a pair of operators: gather, which efficiently aggregates feature responses from a large spatial extent, and excite, which redistributes the pooled information to local features. The operators are cheap, both in terms of number of added parameters and computational complexity, and can be integrated directly in existing architectures to improve their performance. Experiments on several datasets show that gather-excite can bring benefits comparable to increasing the depth of a CNN at a fraction of the cost. For example, we find ResNet-50 with gather-excite operators is able to outperform its 101-layer counterpart on ImageNet with no additional learnable parameters. We also propose a parametric gather-excite operator pair which yields further performance gains, relate it to the recently-introduced Squeeze-and-Excitation Networks, and analyse the effects of these changes to the CNN feature activation statistics.
* [[Look and Think Twice](https://ieeexplore.ieee.org/document/7410695)] <!-- printed -->
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410695)]
    * Title: Look and Think Twice: Capturing Top-Down Visual Attention with Feedback Convolutional Neural Networks
    * Year: 18 February `2016`
    * Authors: Chunshui Cao; Xianming Liu; Yi Yang; Yinan Yu; Jiang Wang; Zilei Wang; Yongzhen Huang; Liang Wang; Chang Huang; Wei Xu; Deva Ramanan; Thomas S. Huang
    * Abstract: While feedforward deep convolutional neural networks (CNNs) have been a great success in computer vision, it is important to note that the human visual cortex generally contains more feedback than feedforward connections. In this paper, we will briefly introduce the background of feedbacks in the human visual cortex, which motivates us to develop a computational feedback mechanism in deep neural networks. In addition to the feedforward inference in traditional neural networks, a feedback loop is introduced to infer the activation status of hidden layer neurons according to the "goal" of the network, e.g., high-level semantic labels. We analogize this mechanism as "Look and Think Twice." The feedback networks help better visualize and understand how deep neural networks work, and capture visual attention on expected objects, even in images with cluttered background and multiple objects. Experiments on ImageNet dataset demonstrate its effectiveness in solving tasks such as image classification and object localization.
* [[Deep Recurrent Attentive Writer (DRAW)](https://arxiv.org/abs/1502.04623)]
    [[pdf](https://arxiv.org/pdf/1502.04623.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1502.04623/)]
    * Title: DRAW: A Recurrent Neural Network For Image Generation
    * Year: 16 Feb `2015`
    * Authors: Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, Daan Wierstra
    * Abstract: This paper introduces the Deep Recurrent Attentive Writer (DRAW) neural network architecture for image generation. DRAW networks combine a novel spatial attention mechanism that mimics the foveation of the human eye, with a sequential variational auto-encoding framework that allows for the iterative construction of complex images. The system substantially improves on the state of the art for generative models on MNIST, and, when trained on the Street View House Numbers dataset, it generates images that cannot be distinguished from real data with the naked eye.
    * Comments:
        * > Gregor et al. [25] employ a differentiable attention model to specify where to read/write image regions for image generation. (Attention to Scale, 2015)
* [[The Application of Two-level Attention Models in Deep Convolutional Neural Network for Fine-grained Image Classification](https://arxiv.org/abs/1411.6447)]
    [[pdf](https://arxiv.org/pdf/1411.6447.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1411.6447/)]
    * Title: The Application of Two-level Attention Models in Deep Convolutional Neural Network for Fine-grained Image Classification
    * Year: 24 Nov `2014`
    * Authors: Tianjun Xiao, Yichong Xu, Kuiyuan Yang, Jiaxing Zhang, Yuxin Peng, Zheng Zhang
    * Abstract: Fine-grained classification is challenging because categories can only be discriminated by subtle and local differences. Variances in the pose, scale or rotation usually make the problem more difficult. Most fine-grained classification systems follow the pipeline of finding foreground object or object parts (where) to extract discriminative features (what). In this paper, we propose to apply visual attention to fine-grained classification task using deep neural network. Our pipeline integrates three types of attention: the bottom-up attention that propose candidate patches, the object-level top-down attention that selects relevant patches to a certain object, and the part-level top-down attention that localizes discriminative parts. We combine these attentions to train domain-specific deep nets, then use it to improve both the what and where aspects. Importantly, we avoid using expensive annotations like bounding box or part information from end-to-end. The weak supervision constraint makes our work easier to generalize. We have verified the effectiveness of the method on the subsets of ILSVRC2012 dataset and CUB200_2011 dataset. Our pipeline delivered significant improvements and achieved the best accuracy under the weakest supervision condition. The performance is competitive against other methods that rely on additional annotations.
* [[BAM](https://arxiv.org/abs/1807.06514)]
    [[pdf](https://arxiv.org/pdf/1807.06514.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1807.06514/)]
    * Title: BAM: Bottleneck Attention Module
    * Year: 17 Jul `2018`
    * Authors: Jongchan Park, Sanghyun Woo, Joon-Young Lee, In So Kweon
    * Abstract: Recent advances in deep neural networks have been developed via architecture search for stronger representational power. In this work, we focus on the effect of attention in general deep neural networks. We propose a simple and effective attention module, named Bottleneck Attention Module (BAM), that can be integrated with any feed-forward convolutional neural networks. Our module infers an attention map along two separate pathways, channel and spatial. We place our module at each bottleneck of models where the downsampling of feature maps occurs. Our module constructs a hierarchical attention at bottlenecks with a number of parameters and it is trainable in an end-to-end manner jointly with any feed-forward models. We validate our BAM through extensive experiments on CIFAR-100, ImageNet-1K, VOC 2007 and MS COCO benchmarks. Our experiments show consistent improvement in classification and detection performances with various models, demonstrating the wide applicability of BAM. The code and models will be publicly available.
* [[CBAM](https://arxiv.org/abs/1807.06521)]
    [[pdf](https://arxiv.org/pdf/1807.06521.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1807.06521/)]
    * Title: CBAM: Convolutional Block Attention Module
    * Year: 17 Jul `2018`
    * Authors: Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
    * Abstract: We propose Convolutional Block Attention Module (CBAM), a simple yet effective attention module for feed-forward convolutional neural networks. Given an intermediate feature map, our module sequentially infers attention maps along two separate dimensions, channel and spatial, then the attention maps are multiplied to the input feature map for adaptive feature refinement. Because CBAM is a lightweight and general module, it can be integrated into any CNN architectures seamlessly with negligible overheads and is end-to-end trainable along with base CNNs. We validate our CBAM through extensive experiments on ImageNet-1K, MS~COCO detection, and VOC~2007 detection datasets. Our experiments show consistent improvements in classification and detection performances with various models, demonstrating the wide applicability of CBAM. The code and models will be publicly available.
* [[Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)]
    [[pdf](https://arxiv.org/pdf/1803.02155.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1803.02155/)]
    * Title: Self-Attention with Relative Position Representations
    * Year: 06 Mar `2018`
    * Authors: Peter Shaw, Jakob Uszkoreit, Ashish Vaswani
    * Abstract: Relying entirely on an attention mechanism, the Transformer introduced by Vaswani et al. (2017) achieves state-of-the-art results for machine translation. In contrast to recurrent and convolutional neural networks, it does not explicitly model relative or absolute position information in its structure. Instead, it requires adding representations of absolute positions to its inputs. In this work we present an alternative approach, extending the self-attention mechanism to efficiently consider representations of the relative positions, or distances between sequence elements. On the WMT 2014 English-to-German and English-to-French translation tasks, this approach yields improvements of 1.3 BLEU and 0.3 BLEU over absolute position representations, respectively. Notably, we observe that combining relative and absolute position representations yields no further improvement in translation quality. We describe an efficient implementation of our method and cast it as an instance of relation-aware self-attention mechanisms that can generalize to arbitrary graph-labeled inputs.
* [[$A^2$-Nets](https://arxiv.org/abs/1810.11579)]
    [[pdf](https://arxiv.org/pdf/1810.11579.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1810.11579/)]
    * Title: $A^2$-Nets: Double Attention Networks
    * Year: 27 Oct `2018`
    * Authors: Yunpeng Chen, Yannis Kalantidis, Jianshu Li, Shuicheng Yan, Jiashi Feng
    * Abstract: Learning to capture long-range relations is fundamental to image/video recognition. Existing CNN models generally rely on increasing depth to model such relations which is highly inefficient. In this work, we propose the "double attention block", a novel component that aggregates and propagates informative global features from the entire spatio-temporal space of input images/videos, enabling subsequent convolution layers to access features from the entire space efficiently. The component is designed with a double attention mechanism in two steps, where the first step gathers features from the entire space into a compact set through second-order attention pooling and the second step adaptively selects and distributes features to each location via another attention. The proposed double attention block is easy to adopt and can be plugged into existing deep neural networks conveniently. We conduct extensive ablation studies and experiments on both image and video recognition tasks for evaluating its performance. On the image recognition task, a ResNet-50 equipped with our double attention blocks outperforms a much larger ResNet-152 architecture on ImageNet-1k dataset with over 40% less the number of parameters and less FLOPs. On the action recognition task, our proposed model achieves the state-of-the-art results on the Kinetics and UCF-101 datasets with significantly higher efficiency than recent works.
* [[Learning what and where to attend](https://arxiv.org/abs/1805.08819)]
    [[pdf](https://arxiv.org/pdf/1805.08819.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1805.08819/)]
    * Title: Learning what and where to attend
    * Year: 22 May `2018`
    * Authors: Drew Linsley, Dan Shiebler, Sven Eberhardt, Thomas Serre
    * Abstract: Most recent gains in visual recognition have originated from the inclusion of attention mechanisms in deep convolutional networks (DCNs). Because these networks are optimized for object recognition, they learn where to attend using only a weak form of supervision derived from image class labels. Here, we demonstrate the benefit of using stronger supervisory signals by teaching DCNs to attend to image regions that humans deem important for object recognition. We first describe a large-scale online experiment (ClickMe) used to supplement ImageNet with nearly half a million human-derived "top-down" attention maps. Using human psychophysics, we confirm that the identified top-down features from ClickMe are more diagnostic than "bottom-up" saliency features for rapid image categorization. As a proof of concept, we extend a state-of-the-art attention network and demonstrate that adding ClickMe supervision significantly improves its accuracy and yields visual features that are more interpretable and more similar to those used by human observers.
* [[Convolutional Self-Attention Networks](https://arxiv.org/abs/1904.03107)]
    [[pdf](https://arxiv.org/pdf/1904.03107.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.03107/)]
    * Title: Convolutional Self-Attention Networks
    * Year: 05 Apr `2019`
    * Authors: Baosong Yang, Longyue Wang, Derek Wong, Lidia S. Chao, Zhaopeng Tu
    * Abstract: Self-attention networks (SANs) have drawn increasing interest due to their high parallelization in computation and flexibility in modeling dependencies. SANs can be further enhanced with multi-head attention by allowing the model to attend to information from different representation subspaces. In this work, we propose novel convolutional self-attention networks, which offer SANs the abilities to 1) strengthen dependencies among neighboring elements, and 2) model the interaction between features extracted by multiple attention heads. Experimental results of machine translation on different language pairs and model settings show that our approach outperforms both the strong Transformer baseline and other existing models on enhancing the locality of SANs. Comparing with prior studies, the proposed model is parameter free in terms of introducing no more parameters.
* [[ConvNeXt](https://arxiv.org/abs/2201.03545)]
    [[pdf](https://arxiv.org/pdf/2201.03545.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2201.03545/)]
    * Title: A ConvNet for the 2020s
    * Year: 10 Jan `2022`
    * Authors: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie
    * Abstract: The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

### Spatial Attention

* DeformableNet
* Vision Transformer
* Point Cloud Transformer
* Swin Transformer

### (2022, SegNeXt) Channel Attention (count=3)

* [[SENet](https://arxiv.org/abs/1709.01507)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1709.01507.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1709.01507/)]
    * Title: Squeeze-and-Excitation Networks
    * Year: 05 Sep `2017`
    * Authors: Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
    * Abstract: The central building block of convolutional neural networks (CNNs) is the convolution operator, which enables networks to construct informative features by fusing both spatial and channel-wise information within local receptive fields at each layer. A broad range of prior research has investigated the spatial component of this relationship, seeking to strengthen the representational power of a CNN by enhancing the quality of spatial encodings throughout its feature hierarchy. In this work, we focus instead on the channel relationship and propose a novel architectural unit, which we term the "Squeeze-and-Excitation" (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels. We show that these blocks can be stacked together to form SENet architectures that generalise extremely effectively across different datasets. We further demonstrate that SE blocks bring significant improvements in performance for existing state-of-the-art CNNs at slight additional computational cost. Squeeze-and-Excitation Networks formed the foundation of our ILSVRC 2017 classification submission which won first place and reduced the top-5 error to 2.251%, surpassing the winning entry of 2016 by a relative improvement of ~25%. Models and code are available at this https URL.
* [[SCANet](https://arxiv.org/abs/1611.05594)]
    [[pdf](https://arxiv.org/pdf/1611.05594.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1611.05594/)]
    * Title: SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning
    * Year: 17 Nov `2016`
    * Authors: Long Chen, Hanwang Zhang, Jun Xiao, Liqiang Nie, Jian Shao, Wei Liu, Tat-Seng Chua
    * Abstract: Visual attention has been successfully applied in structural prediction tasks such as visual captioning and question answering. Existing visual attention models are generally spatial, i.e., the attention is modeled as spatial probabilities that re-weight the last conv-layer feature map of a CNN encoding an input image. However, we argue that such spatial attention does not necessarily conform to the attention mechanism --- a dynamic feature extractor that combines contextual fixations over time, as CNN features are naturally spatial, channel-wise and multi-layer. In this paper, we introduce a novel convolutional neural network dubbed SCA-CNN that incorporates Spatial and Channel-wise Attentions in a CNN. In the task of image captioning, SCA-CNN dynamically modulates the sentence generation context in multi-layer feature maps, encoding where (i.e., attentive spatial locations at multiple layers) and what (i.e., attentive channels) the visual attention is. We evaluate the proposed SCA-CNN architecture on three benchmark image captioning datasets: Flickr8K, Flickr30K, and MSCOCO. It is consistently observed that SCA-CNN significantly outperforms state-of-the-art visual attention-based image captioning methods.
* [[ECANet](https://arxiv.org/abs/1910.03151)]
    [[pdf](https://arxiv.org/pdf/1910.03151.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1910.03151/)]
    * Title: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    * Year: 08 Oct `2019`
    * Authors: Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu
    * Abstract: Recently, channel attention mechanism has demonstrated to offer great potential in improving the performance of deep convolutional neural networks (CNNs). However, most existing methods dedicate to developing more sophisticated attention modules for achieving better performance, which inevitably increase model complexity. To overcome the paradox of performance and complexity trade-off, this paper proposes an Efficient Channel Attention (ECA) module, which only involves a handful of parameters while bringing clear performance gain. By dissecting the channel attention module in SENet, we empirically show avoiding dimensionality reduction is important for learning channel attention, and appropriate cross-channel interaction can preserve performance while significantly decreasing model complexity. Therefore, we propose a local cross-channel interaction strategy without dimensionality reduction, which can be efficiently implemented via $1D$ convolution. Furthermore, we develop a method to adaptively select kernel size of $1D$ convolution, determining coverage of local cross-channel interaction. The proposed ECA module is efficient yet effective, e.g., the parameters and computations of our modules against backbone of ResNet50 are 80 vs. 24.37M and 4.7e-4 GFLOPs vs. 3.86 GFLOPs, respectively, and the performance boost is more than 2% in terms of Top-1 accuracy. We extensively evaluate our ECA module on image classification, object detection and instance segmentation with backbones of ResNets and MobileNetV2. The experimental results show our module is more efficient while performing favorably against its counterparts.

## Light-Weight Networks

### Light-Weight Networks (Others)

* [[SqueezeNext](https://arxiv.org/abs/1803.10615)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1803.10615.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1803.10615/)]
    * Title: SqueezeNext: Hardware-Aware Neural Network Design
    * Year: 23 Mar `2018`
    * Authors: Amir Gholami, Kiseok Kwon, Bichen Wu, Zizheng Tai, Xiangyu Yue, Peter Jin, Sicheng Zhao, Kurt Keutzer
    * Abstract: One of the main barriers for deploying neural networks on embedded systems has been large memory and power consumption of existing neural networks. In this work, we introduce SqueezeNext, a new family of neural network architectures whose design was guided by considering previous architectures such as SqueezeNet, as well as by simulation results on a neural network accelerator. This new network is able to match AlexNet's accuracy on the ImageNet benchmark with $112\times$ fewer parameters, and one of its deeper variants is able to achieve VGG-19 accuracy with only 4.4 Million parameters, ($31\times$ smaller than VGG-19). SqueezeNext also achieves better top-5 classification accuracy with $1.3\times$ fewer parameters as compared to MobileNet, but avoids using depthwise-separable convolutions that are inefficient on some mobile processor platforms. This wide range of accuracy gives the user the ability to make speed-accuracy tradeoffs, depending on the available resources on the target hardware. Using hardware simulation results for power and inference speed on an embedded system has guided us to design variations of the baseline model that are $2.59\times$/$8.26\times$ faster and $2.25\times$/$7.5\times$ more energy efficient as compared to SqueezeNet/AlexNet without any accuracy degradation.
* [[ShiftNet](https://arxiv.org/abs/1711.08141)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1711.08141.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1711.08141/)]
    * Title: Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions
    * Year: 22 Nov `2017`
    * Authors: Bichen Wu, Alvin Wan, Xiangyu Yue, Peter Jin, Sicheng Zhao, Noah Golmant, Amir Gholaminejad, Joseph Gonzalez, Kurt Keutzer
    * Abstract: Neural networks rely on convolutions to aggregate spatial information. However, spatial convolutions are expensive in terms of model size and computation, both of which grow quadratically with respect to kernel size. In this paper, we present a parameter-free, FLOP-free "shift" operation as an alternative to spatial convolutions. We fuse shifts and point-wise convolutions to construct end-to-end trainable shift-based modules, with a hyperparameter characterizing the tradeoff between accuracy and efficiency. To demonstrate the operation's efficacy, we replace ResNet's 3x3 convolutions with shift-based modules for improved CIFAR10 and CIFAR100 accuracy using 60% fewer parameters; we additionally demonstrate the operation's resilience to parameter reduction on ImageNet, outperforming ResNet family members. We finally show the shift operation's applicability across domains, achieving strong performance with fewer parameters on classification, face verification and style transfer.
* [[CondenseNet](https://arxiv.org/abs/1711.09224)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1711.09224.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1711.09224/)]
    * Title: CondenseNet: An Efficient DenseNet using Learned Group Convolutions
    * Year: 25 Nov `2017`
    * Authors: Gao Huang, Shichen Liu, Laurens van der Maaten, Kilian Q. Weinberger
    * Abstract: Deep neural networks are increasingly used on mobile devices, where computational resources are limited. In this paper we develop CondenseNet, a novel network architecture with unprecedented efficiency. It combines dense connectivity with a novel module called learned group convolution. The dense connectivity facilitates feature re-use in the network, whereas learned group convolutions remove connections between layers for which this feature re-use is superfluous. At test time, our model can be implemented using standard group convolutions, allowing for efficient computation in practice. Our experiments show that CondenseNets are far more efficient than state-of-the-art compact convolutional networks such as MobileNets and ShuffleNets.

### (2017, MobileNetV1) Light-Weight Networks (count=7)

> MobileNets are built primarily from depthwise separable convolutions initially introduced in [26] and subsequently used in Inception models [13] to reduce the computation in the first few layers. Flattened networks [16] build a network out of fully factorized convolutions and showed the potential of extremely factorized networks. Independent of this current paper, Factorized Networks [34] introduces a similar factorized convolution as well as the use of topological connections. Subsequently, the Xception network [3] demonstrated how to scale up depthwise separable filters to out perform Inception V3 networks. Another small network is Squeezenet [12] which uses a bottleneck approach to design a very small network. Other reduced computation networks include structured transform networks [28] and deep fried convnets [37].

* Inception Networks
* [[Flattened Networks](https://arxiv.org/abs/1412.5474)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1412.5474.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.5474/)]
    * Title: Flattened Convolutional Neural Networks for Feedforward Acceleration
    * Year: 17 Dec `2014`
    * Authors: Jonghoon Jin, Aysegul Dundar, Eugenio Culurciello
    * Abstract: We present flattened convolutional neural networks that are designed for fast feedforward execution. The redundancy of the parameters, especially weights of the convolutional filters in convolutional neural networks has been extensively studied and different heuristics have been proposed to construct a low rank basis of the filters after training. In this work, we train flattened networks that consist of consecutive sequence of one-dimensional filters across all directions in 3D space to obtain comparable performance as conventional convolutional networks. We tested flattened model on different datasets and found that the flattened layer can effectively substitute for the 3D filters without loss of accuracy. The flattened convolution pipelines provide around two times speed-up during feedforward pass compared to the baseline model due to the significant reduction of learning parameters. Furthermore, the proposed method does not require efforts in manual tuning or post processing once the model is trained.
    * Comments:
        * > (2017, MobileNetV1) Flattened networks [16] build a network out of fully factorized convolutions and showed the potential of extremely factorized networks.
* [[Factorized Networks](https://arxiv.org/abs/1608.04337)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1608.04337.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1608.04337/)]
    * Title: Design of Efficient Convolutional Layers using Single Intra-channel Convolution, Topological Subdivisioning and Spatial "Bottleneck" Structure
    * Year: 15 Aug `2016`
    * Authors: Min Wang, Baoyuan Liu, Hassan Foroosh
    * Institutions: [Department of EECS, University of Central Florida]
    * Abstract: Deep convolutional neural networks achieve remarkable visual recognition performance, at the cost of high computational complexity. In this paper, we have a new design of efficient convolutional layers based on three schemes. The 3D convolution operation in a convolutional layer can be considered as performing spatial convolution in each channel and linear projection across channels simultaneously. By unravelling them and arranging the spatial convolution sequentially, the proposed layer is composed of a single intra-channel convolution, of which the computation is negligible, and a linear channel projection. A topological subdivisioning is adopted to reduce the connection between the input channels and output channels. Additionally, we also introduce a spatial "bottleneck" structure that utilizes a convolution-projection-deconvolution pipeline to take advantage of the correlation between adjacent pixels in the input. Our experiments demonstrate that the proposed layers remarkably outperform the standard convolutional layers with regard to accuracy/complexity ratio. Our models achieve similar accuracy to VGG, ResNet-50, ResNet-101 while requiring 42, 4.5, 6.5 times less computation respectively.
    * Comments:
        * > (2017, MobileNetV1) Independent of this current paper, Factorized Networks [34] introduces a similar factorized convolution as well as the use of topological connections.
* [[Xception Networks](https://arxiv.org/abs/1610.02357)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1610.02357.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1610.02357/)]
    * Title: Xception: Deep Learning with Depthwise Separable Convolutions
    * Year: 07 Oct `2016`
    * Authors: François Chollet
    * Institution: [Google, Inc.]
    * Abstract: We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.
    * Comments:
        * > (2017, MobileNetV1) Subsequently, the Xception network [3] demonstrated how to scale up depthwise separable filters to out perform Inception V3 networks.
        * > (2017, ShuffleNetV1) Depthwise separable convolution proposed in Xception [3] generalizes the ideas of separable convolutions in Inception series [34, 32].
        * (2017, Transformer) dilated convolutions.
* [[SqueezeNet](https://arxiv.org/abs/1602.07360)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1602.07360.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1602.07360/)]
    * Title: SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
    * Year: 24 Feb `2016`
    * Authors: Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
    * Abstract: Recent research on deep neural networks has focused primarily on improving accuracy. For a given accuracy level, it is typically possible to identify multiple DNN architectures that achieve that accuracy level. With equivalent accuracy, smaller DNN architectures offer at least three advantages: (1) Smaller DNNs require less communication across servers during distributed training. (2) Smaller DNNs require less bandwidth to export a new model from the cloud to an autonomous car. (3) Smaller DNNs are more feasible to deploy on FPGAs and other hardware with limited memory. To provide all of these advantages, we propose a small DNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet). The SqueezeNet architecture is available for download here: this https URL
    * Comments:
        * > (2017, MobileNetV1) Another small network is Squeezenet [12] which uses a bottleneck approach to design a very small network.
* [[Structured Transform Networks](https://arxiv.org/abs/1510.01722)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1510.01722.pdf)]
    [vanity]
    * Title: Structured Transforms for Small-Footprint Deep Learning
    * Year: 06 Oct `2015`
    * Authors: Vikas Sindhwani, Tara N. Sainath, Sanjiv Kumar
    * Institutions: [Google, New York]
    * Abstract: We consider the task of building compact deep learning pipelines suitable for deployment on storage and power constrained mobile devices. We propose a unified framework to learn a broad family of structured parameter matrices that are characterized by the notion of low displacement rank. Our structured transforms admit fast function and gradient evaluation, and span a rich range of parameter sharing configurations whose statistical modeling capacity can be explicitly tuned along a continuum from structured to unstructured. Experimental results show that these transforms can significantly accelerate inference and forward/backward passes during training, and offer superior accuracy-compactness-speed tradeoffs in comparison to a number of existing techniques. In keyword spotting applications in mobile speech recognition, our methods are much more effective than standard linear low-rank bottleneck layers and nearly retain the performance of state of the art models, while providing more than 3.5-fold compression.
* [[Deep Fried Convnets](https://arxiv.org/abs/1412.7149)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1412.7149.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.7149/)]
    * Title: Deep Fried Convnets
    * Year: 22 Dec `2014`
    * Authors: Zichao Yang, Marcin Moczulski, Misha Denil, Nando de Freitas, Alex Smola, Le Song, Ziyu Wang
    * Institutions: [Carnegie Mellon University], [University of Oxford], [Georgia Institute of Technology], [Google], [Google DeepMind], [Canadian Institute for Advanced Research]
    * Abstract: The fully connected layers of a deep convolutional neural network typically contain over 90% of the network parameters, and consume the majority of the memory required to store the network parameters. Reducing the number of parameters while preserving essentially the same predictive performance is critically important for operating deep neural networks in memory constrained environments such as GPUs or embedded devices. In this paper we show how kernel methods, in particular a single Fastfood layer, can be used to replace all fully connected layers in a deep convolutional neural network. This novel Fastfood layer is also end-to-end trainable in conjunction with convolutional layers, allowing us to combine them into a new architecture, named deep fried convolutional networks, which substantially reduces the memory footprint of convolutional networks trained on MNIST and ImageNet with no drop in predictive performance.

### (2021, MobileViTv1) Light-Weight Networks (count=3+5)

> The basic building layer in CNNs is a standard convolutional layer. Because this layer is computationally expensive, several factorization-based methods have been proposed to make it light-weight and mobile-friendly (e.g., Jin et al., 2014; Chollet, 2017; Mehta et al., 2020). Of these, separable convolutions of Chollet (2017) have gained interest, and are widely used across state-of-the-art light-weight CNNs for mobile vision tasks, including MobileNets (Howard et al., 2017; Sandler et al., 2018; Howard et al., 2019), ShuffleNetv2 (Ma et al., 2018), ESPNetv2 (Mehta et al., 2019), MixNet (Tan & Le, 2019b), and MNASNet (Tan et al., 2019). These light-weight CNNs are versatile and easy to train. For example, these networks can easily replace the heavy-weight backbones (e.g., ResNet) in existing task-specific models (e.g., DeepLabv3) to reduce the network size and improve latency. Despite these benefits, one major drawback of these methods is that they are spatially local. (MobileViTv1, 2021)

* Flattened Networks
* Xception Networks
* [[DiCENet](https://arxiv.org/abs/1906.03516)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1906.03516.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1906.03516/)]
    * Title: DiCENet: Dimension-wise Convolutions for Efficient Networks
    * Year: 08 Jun `2019`
    * Authors: Sachin Mehta, Hannaneh Hajishirzi, Mohammad Rastegari
    * Institutions: [University of Washington]
    * Abstract: We introduce a novel and generic convolutional unit, DiCE unit, that is built using dimension-wise convolutions and dimension-wise fusion. The dimension-wise convolutions apply light-weight convolutional filtering across each dimension of the input tensor while dimension-wise fusion efficiently combines these dimension-wise representations; allowing the DiCE unit to efficiently encode spatial and channel-wise information contained in the input tensor. The DiCE unit is simple and can be seamlessly integrated with any architecture to improve its efficiency and performance. Compared to depth-wise separable convolutions, the DiCE unit shows significant improvements across different architectures. When DiCE units are stacked to build the DiCENet model, we observe significant improvements over state-of-the-art models across various computer vision tasks including image classification, object detection, and semantic segmentation. On the ImageNet dataset, the DiCENet delivers 2-4% higher accuracy than state-of-the-art manually designed models (e.g., MobileNetv2 and ShuffleNetv2). Also, DiCENet generalizes better to tasks (e.g., object detection) that are often used in resource-constrained devices in comparison to state-of-the-art separable convolution-based efficient networks, including neural search-based methods (e.g., MobileNetv3 and MixNet. Our source code in PyTorch is open-source and is available at this https URL

* MobileNetV1, MobileNetV2, MobileNetV3.
* ShuffleNetV2
* ESPNetV2
* [[Mixconv](https://arxiv.org/abs/1907.09595)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1907.09595)]
    [[vanity](https://www.arxiv-vanity.com/papers/1907.09595/)]
    * Title: MixConv: Mixed Depthwise Convolutional Kernels
    * Year: 22 Jul `2019`
    * Authors: Mingxing Tan, Quoc V. Le
    * Institutions: [Google Brain]
    * Abstract: Depthwise convolution is becoming increasingly popular in modern efficient ConvNets, but its kernel size is often overlooked. In this paper, we systematically study the impact of different kernel sizes, and observe that combining the benefits of multiple kernel sizes can lead to better accuracy and efficiency. Based on this observation, we propose a new mixed depthwise convolution (MixConv), which naturally mixes up multiple kernel sizes in a single convolution. As a simple drop-in replacement of vanilla depthwise convolution, our MixConv improves the accuracy and efficiency for existing MobileNets on both ImageNet classification and COCO object detection. To demonstrate the effectiveness of MixConv, we integrate it into AutoML search space and develop a new family of models, named as MixNets, which outperform previous mobile models including MobileNetV2 [20] (ImageNet top-1 accuracy +4.2%), ShuffleNetV2 [16] (+3.5%), MnasNet [26] (+1.3%), ProxylessNAS [2] (+2.2%), and FBNet [27] (+2.0%). In particular, our MixNet-L achieves a new state-of-the-art 78.9% ImageNet top-1 accuracy under typical mobile settings (<600M FLOPS). Code is at this https URL tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
* MnasNet (see neural_architecture_search.md)

### MobileNet Series (count=3+2)

* [[MobileNetV1](https://arxiv.org/abs/1704.04861)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1704.04861.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1704.04861/)]
    * Title: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    * Year: 17 Apr `2017`
    * Authors: Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
    * Institutions: [Google Inc.]
    * Abstract: We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.
* [[MobileNetV2](https://arxiv.org/abs/1801.04381)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1801.04381.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1801.04381/)]
    * Title: MobileNetV2: Inverted Residuals and Linear Bottlenecks
    * Year: 13 Jan `2018`
    * Authors: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    * Institutions: [Google Inc.]
    * Abstract: In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3. The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters
* [[MobileNetV3](https://arxiv.org/abs/1905.02244)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1905.02244.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1905.02244/)]
    * Title: Searching for MobileNetV3
    * Year: 06 May `2019`
    * Authors: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
    * Abstract: We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2\% more accurate on ImageNet classification while reducing latency by 15\% compared to MobileNetV2. MobileNetV3-Small is 4.6\% more accurate while reducing latency by 5\% compared to MobileNetV2. MobileNetV3-Large detection is 25\% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30\% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.
* [[Swish](https://arxiv.org/abs/1702.03118)]
    [[pdf](https://arxiv.org/pdf/1702.03118.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1702.03118/)]
    * Title: Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning
    * Year: 10 Feb `2017`
    * Authors: Stefan Elfwing, Eiji Uchibe, Kenji Doya
    * Abstract: In recent years, neural networks have enjoyed a renaissance as function approximators in reinforcement learning. Two decades after Tesauro's TD-Gammon achieved near top-level human performance in backgammon, the deep reinforcement learning algorithm DQN achieved human-level performance in many Atari 2600 games. The purpose of this study is twofold. First, we propose two activation functions for neural network function approximation in reinforcement learning: the sigmoid-weighted linear unit (SiLU) and its derivative function (dSiLU). The activation of the SiLU is computed by the sigmoid function multiplied by its input. Second, we suggest that the more traditional approach of using on-policy learning with eligibility traces, instead of experience replay, and softmax action selection with simple annealing can be competitive with DQN, without the need for a separate target network. We validate our proposed approach by, first, achieving new state-of-the-art results in both stochastic SZ-Tetris and Tetris with a small 10$\times$10 board, using TD($\lambda$) learning and shallow dSiLU network agents, and, then, by outperforming DQN in the Atari 2600 domain by using a deep Sarsa($\lambda$) agent with SiLU and dSiLU hidden units.

### EfficientNets (count=3)

* [[EfficientNetV1](https://arxiv.org/abs/1905.11946)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1905.11946.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1905.11946/)]
    * Title: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    * Year: 28 May `2019`
    * Authors: Mingxing Tan, Quoc V. Le
    * Institutions: [Google Research, Brain Team]
    * Abstract: Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet. To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at this https URL.
    * Comments:
        * > Our study shows in EfficientNets: (1) training with very large image sizes is slow; (2) depthwise convolutions are slow in early layers. (3) equally scaling up every stage is sub-optimal. (EfficientNetV2, 2021)
        * > Many works, such as DenseNet (Huang et al., 2017) and EfficientNet (Tan & Le, 2019a), focus on parameter efficiency, aiming to achieve better accuracy with less parameters. (EfficientNetV2, 2021)
* [[EfficientNetV2](https://arxiv.org/abs/2104.00298)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2104.00298.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2104.00298/)]
    * Title: EfficientNetV2: Smaller Models and Faster Training
    * Year: 01 Apr `2021`
    * Authors: Mingxing Tan, Quoc V. Le
    * Institutions: [Google Research, Brain Team]
    * Abstract: This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. To develop this family of models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller. Our training can be further sped up by progressively increasing the image size during training, but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose to adaptively adjust regularization (e.g., dropout and data augmentation) as well, such that we can achieve both fast training and good accuracy. With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources. Code will be available at this https URL.
* [EfficientNet-X](https://arxiv.org/abs/2102.05610)
    [[pdf](https://arxiv.org/pdf/2102.05610.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2102.05610/)]
    * Title: Searching for Fast Model Families on Datacenter Accelerators
    * Year: 10 Feb `2021`
    * Authors: Sheng Li, Mingxing Tan, Ruoming Pang, Andrew Li, Liqun Cheng, Quoc Le, Norman P. Jouppi
    * Abstract: Neural Architecture Search (NAS), together with model scaling, has shown remarkable progress in designing high accuracy and fast convolutional architecture families. However, as neither NAS nor model scaling considers sufficient hardware architecture details, they do not take full advantage of the emerging datacenter (DC) accelerators. In this paper, we search for fast and accurate CNN model families for efficient inference on DC accelerators. We first analyze DC accelerators and find that existing CNNs suffer from insufficient operational intensity, parallelism, and execution efficiency. These insights let us create a DC-accelerator-optimized search space, with space-to-depth, space-to-batch, hybrid fused convolution structures with vanilla and depthwise convolutions, and block-wise activation functions. On top of our DC accelerator optimized neural architecture search space, we further propose a latency-aware compound scaling (LACS), the first multi-objective compound scaling method optimizing both accuracy and latency. Our LACS discovers that network depth should grow much faster than image size and network width, which is quite different from previous compound scaling results. With the new search space and LACS, our search and scaling on datacenter accelerators results in a new model series named EfficientNet-X. EfficientNet-X is up to more than 2X faster than EfficientNet (a model series with state-of-the-art trade-off on FLOPs and accuracy) on TPUv3 and GPUv100, with comparable accuracy. EfficientNet-X is also up to 7X faster than recent RegNet and ResNeSt on TPUv3 and GPUv100.
    * Comments:
        * > RegNet (Radosavovic et al., 2020), ResNeSt (Zhang et al., 2020), TResNet (Ridnik et al., 2020), and EfficientNet-X (Li et al., 2021) focus on GPU and/or TPU inference speed. (EfficientNetV2, 2021)

### ShuffleNets (count=2)

* [[ShuffleNetV1](https://arxiv.org/abs/1707.01083)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1707.01083.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1707.01083/)]
    * Title: ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    * Year: 04 Jul `2017`
    * Authors: Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
    * Abstract: We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet on ImageNet classification task, under the computation budget of 40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves ~13x actual speedup over AlexNet while maintaining comparable accuracy.
    * Comments:
        * > (2018, MobileNetV2) Depthwise Separable Convolutions are a key building block for many efficient neural network architectures [26, 27, 19].
        * > (2018, MobileNetV2) ShuffleNet uses Group Convolutions [19] and shuffling, it also uses conventional residual approach where inner blocks are narrower than output.
        * > The ShuffleNet module [17], shown in Fig. 3b, is based on the principle of reduce-transform-expand. It is an optimized version of the bottleneck block in ResNet [47]. (ESPNetV1, 2018)
* [[ShuffleNetV2](https://arxiv.org/abs/1807.11164)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1807.11164.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1807.11164/)]
    * Title: ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    * Year: 30 Jul `2018`
    * Authors: Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    * Abstract: Currently, the neural network architecture design is mostly guided by the \emph{indirect} metric of computation complexity, i.e., FLOPs. However, the \emph{direct} metric, e.g., speed, also depends on the other factors such as memory access cost and platform characterics. Thus, this work proposes to evaluate the direct metric on the target platform, beyond only considering FLOPs. Based on a series of controlled experiments, this work derives several practical \emph{guidelines} for efficient network design. Accordingly, a new architecture is presented, called \emph{ShuffleNet V2}. Comprehensive ablation experiments verify that our model is the state-of-the-art in terms of speed and accuracy tradeoff.
    * Comments:
        * > (2018, ESPNetV2) In addition to convolutional factorization, a network's efficiency and accuracy can be further improved using methods such as channel shuffle [29] and channel split [29].

## Information Bottleneck

* [[Deep VIB](https://arxiv.org/abs/1612.00410)]
    [[pdf](https://arxiv.org/pdf/1612.00410.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1612.00410/)]
    * Title: Deep Variational Information Bottleneck
    * Year: 01 Dec `2016`
    * Authors: Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy
    * Abstract: We present a variational approximation to the information bottleneck of Tishby et al. (1999). This variational approach allows us to parameterize the information bottleneck model using a neural network and leverage the reparameterization trick for efficient training. We call this method "Deep Variational Information Bottleneck", or Deep VIB. We show that models trained with the VIB objective outperform those that are trained with other forms of regularization, in terms of generalization performance and robustness to adversarial attack.
* [[Relevant sparse codes with variational information bottleneck](https://arxiv.org/abs/1605.07332)]
    [[pdf](https://arxiv.org/pdf/1605.07332.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.07332/)]
    * Title: Relevant sparse codes with variational information bottleneck
    * Year: 24 May `2016`
    * Authors: Matthew Chalk, Olivier Marre, Gasper Tkacik
    * Abstract: In many applications, it is desirable to extract only the relevant aspects of data. A principled way to do this is the information bottleneck (IB) method, where one seeks a code that maximizes information about a 'relevance' variable, Y, while constraining the information encoded about the original data, X. Unfortunately however, the IB method is computationally demanding when data are high-dimensional and/or non-gaussian. Here we propose an approximate variational scheme for maximizing a lower bound on the IB objective, analogous to variational EM. Using this method, we derive an IB algorithm to recover features that are both relevant and sparse. Finally, we demonstrate how kernelized versions of the algorithm can be used to address a broad range of problems with non-linear relation between X and Y.
* [[InfoBot](https://arxiv.org/abs/1901.10902)]
    [[pdf](https://arxiv.org/pdf/1901.10902.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1901.10902/)]
    * Title: InfoBot: Transfer and Exploration via the Information Bottleneck
    * Year: 30 Jan `2019`
    * Authors: Anirudh Goyal, Riashat Islam, Daniel Strouse, Zafarali Ahmed, Matthew Botvinick, Hugo Larochelle, Yoshua Bengio, Sergey Levine
    * Abstract: A central challenge in reinforcement learning is discovering effective policies for tasks where rewards are sparsely distributed. We postulate that in the absence of useful reward signals, an effective exploration strategy should seek out {\it decision states}. These states lie at critical junctions in the state space from where the agent can transition to new, potentially unexplored regions. We propose to learn about decision states from prior experience. By training a goal-conditioned policy with an information bottleneck, we can identify decision states by examining where the model actually leverages the goal state. We find that this simple mechanism effectively identifies decision states, even in partially observed settings. In effect, the model learns the sensory cues that correlate with potential subgoals. In new environments, this model can then identify novel subgoals for further exploration, guiding the agent through a sequence of potential decision states and through new regions of the state space.

## Model Compression

### (2016, RexNeXt) Compressing Convolutional Networks (count=4)

* [[Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](https://arxiv.org/abs/1404.0736)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1404.0736.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1404.0736/)]
    * Title: Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation
    * Year: 02 Apr `2014`
    * Authors: Emily Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, Rob Fergus
    * Abstract: We present techniques for speeding up the test-time evaluation of large convolutional networks, designed for object recognition tasks. These models deliver impressive accuracy but each image evaluation requires millions of floating point operations, making their deployment on smartphones and Internet-scale clusters problematic. The computation is dominated by the convolution operations in the lower layers of the model. We exploit the linear structure present within the convolutional filters to derive approximations that significantly reduce the required computation. Using large state-of-the-art models, we demonstrate we demonstrate speedups of convolutional layers on both CPU and GPU by a factor of 2x, while keeping the accuracy within 1% of the original model.
* Speeding up Convolutional Neural Networks with Low Rank Expansions
* Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications
* Deep roots

### (ESPNetV1, 2018) Compressing Convolutional Networks (count=3)

* [[Hashing](https://arxiv.org/abs/1504.04788)]
    [[pdf](https://arxiv.org/pdf/1504.04788.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1504.04788/)]
    * Title: Compressing Neural Networks with the Hashing Trick
    * Year: 19 Apr `2015`
    * Authors: Wenlin Chen, James T. Wilson, Stephen Tyree, Kilian Q. Weinberger, Yixin Chen
    * Abstract: As deep nets are increasingly used in applications suited for mobile devices, a fundamental dilemma becomes apparent: the trend in deep learning is to grow models to absorb ever-increasing data set sizes; however mobile devices are designed with very little memory and cannot store such large models. We present a novel network architecture, HashedNets, that exploits inherent redundancy in neural networks to achieve drastic reductions in model sizes. HashedNets uses a low-cost hash function to randomly group connection weights into hash buckets, and all connections within the same hash bucket share a single parameter value. These parameters are tuned to adjust to the HashedNets weight sharing architecture with standard backprop during training. Our hashing procedure introduces no additional memory overhead, and we demonstrate on several benchmark data sets that HashedNets shrink the storage requirements of neural networks substantially while mostly preserving generalization performance.
* [[Pruning](https://arxiv.org/abs/1510.00149)]
    [[pdf](https://arxiv.org/pdf/1510.00149.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1510.00149/)]
    * Title: Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding
    * Year: 01 Oct `2015`
    * Authors: Song Han, Huizi Mao, William J. Dally
    * Abstract: Neural networks are both computationally intensive and memory intensive, making them difficult to deploy on embedded systems with limited hardware resources. To address this limitation, we introduce "deep compression", a three stage pipeline: pruning, trained quantization and Huffman coding, that work together to reduce the storage requirement of neural networks by 35x to 49x without affecting their accuracy. Our method first prunes the network by learning only the important connections. Next, we quantize the weights to enforce weight sharing, finally, we apply Huffman coding. After the first two steps we retrain the network to fine tune the remaining connections and the quantized centroids. Pruning, reduces the number of connections by 9x to 13x; Quantization then reduces the number of bits that represent each connection from 32 to 5. On the ImageNet dataset, our method reduced the storage required by AlexNet by 35x, from 240MB to 6.9MB, without loss of accuracy. Our method reduced the size of VGG-16 by 49x from 552MB to 11.3MB, again with no loss of accuracy. This allows fitting the model into on-chip SRAM cache rather than off-chip DRAM memory. Our compression method also facilitates the use of complex neural networks in mobile applications where application size and download bandwidth are constrained. Benchmarked on CPU, GPU and mobile GPU, compressed network has 3x to 4x layerwise speedup and 3x to 7x better energy efficiency.
* [[Vector Quantization](https://arxiv.org/abs/1512.06473)]
    [[pdf](https://arxiv.org/pdf/1512.06473.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1512.06473/)]
    * Title: Quantized Convolutional Neural Networks for Mobile Devices
    * Year: 21 Dec `2015`
    * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, Jian Cheng
    * Abstract: Recently, convolutional neural networks (CNN) have demonstrated impressive performance in various computer vision tasks. However, high performance hardware is typically indispensable for the application of CNN models due to the high computation complexity, which prohibits their further extensions. In this paper, we propose an efficient framework, namely Quantized CNN, to simultaneously speed-up the computation and reduce the storage and memory overhead of CNN models. Both filter kernels in convolutional layers and weighting matrices in fully-connected layers are quantized, aiming at minimizing the estimation error of each layer's response. Extensive experiments on the ILSVRC-12 benchmark demonstrate 4~6x speed-up and 15~20x compression with merely one percentage loss of classification accuracy. With our quantized CNN model, even mobile devices can accurately classify images within one second.

### (ESPNetV2, 2018) Compressing Convolutional Networks (count=2)

* [Constrained Optimization Based Low-Rank Approximation of Deep Neural Networks](https://openaccess.thecvf.com/content_ECCV_2018/html/Chong_Li_Constrained_Optimization_Based_ECCV_2018_paper.html)
    * Title: Constrained Optimization Based Low-Rank Approximation of Deep Neural Networks
    * Year: `2018`
    * Authors: Chong Li, C. J. Richard Shi
    * Abstract: We present COBLA---Constrained Optimization Based Low-rank Approximation---a systematic method of finding an optimal low-rank approximation of a trained convolutional neural network, subject to constraints in the number of multiply-accumulate (MAC) operations and the memory footprint. COBLA optimally allocates the constrained computation resource into each layer of the approximated network. The singular value decomposition of the network weight is computed, then a binary masking variable is introduced to denote whether a particular singular value and the corresponding singular vectors are used in low-rank approximation. With this formulation, the number of the MAC operations and the memory footprint are represented as linear constraints in terms of the binary masking variables. The resulted 0-1 integer programming problem is approximately solved by sequential quadratic programming. COBLA does not introduce any hyperparameter. We empirically demonstrate that COBLA outperforms prior art using the SqueezeNet and VGG-16 architecture on the ImageNet dataset.
* Learning Structured Sparsity in Deep Neural Networks

### Low-bit Networks (2017, MobileNetV1) (count=3)

* [[Training deep neural networks with low precision multiplications](https://arxiv.org/abs/1412.7024)]
    [[pdf](https://arxiv.org/pdf/1412.7024.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.7024/)]
    * Title: Training deep neural networks with low precision multiplications
    * Year: 22 Dec `2014`
    * Authors: Matthieu Courbariaux, Yoshua Bengio, Jean-Pierre David
    * Institutions: [École Polytechnique de Montréal], [Université de Montréal, CIFAR Senior Fellow]
    * Abstract: Multipliers are the most space and power-hungry arithmetic operators of the digital implementation of deep neural networks. We train a set of state-of-the-art neural networks (Maxout networks) on three benchmark datasets: MNIST, CIFAR-10 and SVHN. They are trained with three distinct formats: floating point, fixed point and dynamic fixed point. For each of those datasets and for each of those formats, we assess the impact of the precision of the multiplications on the final error after training. We find that very low precision is sufficient not just for running trained networks but also for training them. For example, it is possible to train Maxout networks with 10 bits multiplications.
* [[XNOR-Net](https://arxiv.org/abs/1603.05279)]
    [[pdf](https://arxiv.org/pdf/1603.05279.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1603.05279/)]
    * Title: XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
    * Year: 16 Mar `2016`
    * Authors: Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi
    * Abstract: We propose two efficient approximations to standard convolutional neural networks: Binary-Weight-Networks and XNOR-Networks. In Binary-Weight-Networks, the filters are approximated with binary values resulting in 32x memory saving. In XNOR-Networks, both the filters and the input to convolutional layers are binary. XNOR-Networks approximate convolutions using primarily binary operations. This results in 58x faster convolutional operations and 32x memory savings. XNOR-Nets offer the possibility of running state-of-the-art networks on CPUs (rather than GPUs) in real-time. Our binary networks are simple, accurate, efficient, and work on challenging visual tasks. We evaluate our approach on the ImageNet classification task. The classification accuracy with a Binary-Weight-Network version of AlexNet is only 2.9% less than the full-precision AlexNet (in top-1 measure). We compare our method with recent network binarization methods, BinaryConnect and BinaryNets, and outperform these methods by large margins on ImageNet, more than 16% in top-1 accuracy.
* [[Quantized Neural Networks](https://arxiv.org/abs/1609.07061)]
    [[pdf](https://arxiv.org/pdf/1609.07061.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1609.07061/)]
    * Title: Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations
    * Year: 22 Sep `2016`
    * Authors: Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio
    * Abstract: We introduce a method to train Quantized Neural Networks (QNNs) --- neural networks with extremely low precision (e.g., 1-bit) weights and activations, at run-time. At train-time the quantized weights and activations are used for computing the parameter gradients. During the forward pass, QNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations. As a result, power consumption is expected to be drastically reduced. We trained QNNs over the MNIST, CIFAR-10, SVHN and ImageNet datasets. The resulting QNNs achieve prediction accuracy comparable to their 32-bit counterparts. For example, our quantized version of AlexNet with 1-bit weights and 2-bit activations achieves $51\%$ top-1 accuracy. Moreover, we quantize the parameter gradients to 6-bits as well which enables gradients computation using only bit-wise operation. Quantized recurrent neural networks were tested over the Penn Treebank dataset, and achieved comparable accuracy as their 32-bit counterparts using only 4-bits. Last but not least, we programmed a binary matrix multiplication GPU kernel with which it is possible to run our MNIST QNN 7 times faster than with an unoptimized GPU kernel, without suffering any loss in classification accuracy. The QNN code is available online.

### Low-bit Networks (2018, ESPNetV1) (4)

* XNOR-Net
* Quantized Neural Networks
* [Fixed-point Networks](https://ieeexplore.ieee.org/document/6986082)
    * Title: Fixed-point feedforward deep neural network design using weights +1, 0, and −1
    * Year: 18 December `2014`
    * Authors: Kyuyeon Hwang; Wonyong Sung
    * Abstract: Feedforward deep neural networks that employ multiple hidden layers show high performance in many applications, but they demand complex hardware for implementation. The hardware complexity can be much lowered by minimizing the word-length of weights and signals, but direct quantization for fixed-point network design does not yield good results. We optimize the fixed-point design by employing backpropagation based retraining. The designed fixed-point networks with ternary weights (+1, 0, and -1) and 3-bit signal show only negligible performance loss when compared to the floating-point coun-terparts. The backpropagation for retraining uses quantized weights and fixed-point signal to compute the output, but utilizes high precision values for adapting the networks. A character recognition and a phoneme recognition examples are presented.
* [[Binarized Networks](https://arxiv.org/abs/1602.02830)]
    [[pdf](https://arxiv.org/pdf/1602.02830.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1602.02830/)]
    * Title: Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1
    * Year: 09 Feb `2016`
    * Authors: Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio
    * Abstract: We introduce a method to train Binarized Neural Networks (BNNs) - neural networks with binary weights and activations at run-time. At training-time the binary weights and activations are used for computing the parameters gradients. During the forward pass, BNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations, which is expected to substantially improve power-efficiency. To validate the effectiveness of BNNs we conduct two sets of experiments on the Torch7 and Theano frameworks. On both, BNNs achieved nearly state-of-the-art results over the MNIST, CIFAR-10 and SVHN datasets. Last but not least, we wrote a binary matrix multiplication GPU kernel with which it is possible to run our MNIST BNN 7 times faster than with an unoptimized GPU kernel, without suffering any loss in classification accuracy. The code for training and running our BNNs is available on-line.

### Low-bit Networks (2018, ESPNetV2) (3)

* XNOR-Net
* Quantized Neural Networks
* [Expectation Backpropagation](https://proceedings.neurips.cc/paper/2014/hash/076a0c97d09cf1a0ec3e19c7f2529f2b-Abstract.html)
    * Title: Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights
    * Year: `2014`
    * Authors: Daniel Soudry, Itay Hubara, Ron Meir
    * Abstract: Multilayer Neural Networks (MNNs) are commonly trained using gradient descent-based methods, such as BackPropagation (BP). Inference in probabilistic graphical models is often done using variational Bayes methods, such as Expectation Propagation (EP). We show how an EP based approach can also be used to train deterministic MNNs. Specifically, we approximate the posterior of the weights given the data using a “mean-field” factorized distribution, in an online setting. Using online EP and the central limit theorem we find an analytical approximation to the Bayes update of this posterior, as well as the resulting Bayes estimates of the weights and outputs. Despite a different origin, the resulting algorithm, Expectation BackPropagation (EBP), is very similar to BP in form and efficiency. However, it has several additional advantages: (1) Training is parameter-free, given initial conditions (prior) and the MNN architecture. This is useful for large-scale problems, where parameter tuning is a major challenge. (2) The weights can be restricted to have discrete values. This is especially useful for implementing trained MNNs in precision limited hardware chips, thus improving their speed and energy efficiency by several orders of magnitude. We test the EBP algorithm numerically in eight binary text classification tasks. In all tasks, EBP outperforms: (1) standard BP with the optimal constant learning rate (2) previously reported state of the art. Interestingly, EBP-trained MNNs with binary weights usually perform better than MNNs with continuous (real) weights - if we average the MNN output using the inferred posterior.

## Low-Rank Approximations

### Low-Rank Approximations

* [Deformable kernels for early vision](https://ieeexplore.ieee.org/document/391394)
    * Title: Deformable kernels for early vision
    * Year: May `1995`
    * Author: P. Perona
    * Abstract: Early vision algorithms often have a first stage of linear-filtering that 'extracts' from the image information at multiple scales of resolution and multiple orientations. A common difficulty in the design and implementation of such schemes is that one feels compelled to discretize coarsely the space of scales and orientations in order to reduce computation and storage costs. A technique is presented that allows: 1) computing the best approximation of a given family using linear combinations of a small number of 'basis' functions; and 2) describing all finite-dimensional families, i.e., the families of filters for which a finite dimensional representation is possible with no error. The technique is based on singular value decomposition and may be applied to generating filters in arbitrary dimensions and subject to arbitrary deformations. The relevant functional analysis results are reviewed and precise conditions for the decomposition to be feasible are stated. Experimental results are presented that demonstrate the applicability of the technique to generating multiorientation multi-scale 2D edge-detection kernels. The implementation issues are also discussed.
* [[A rank minimization heuristic with application to minimum order system approximation](https://ieeexplore.ieee.org/document/945730)] <!-- printed -->
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=945730)]
    * Title: A rank minimization heuristic with application to minimum order system approximation
    * Year: 07 August `2002`
    * Authors: M. Fazel; H. Hindi; S.P. Boyd
    * Institutions: [Stanford University]
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
* Exploiting Linear Structure

### Low-Rank Approximations (2016, Deep Roots) (count=3+3)

* [Simplifying convnets for fast learning](https://dl.acm.org/doi/10.1007/978-3-642-33266-1_8)
    * Title: Simplifying convnets for fast learning
    * Year: 11 September `2012`
    * Authors: Franck Mamalet, Christophe Garcia
    * Abstract: In this paper, we propose different strategies for simplifying filters, used as feature extractors, to be learnt in convolutional neural networks (ConvNets) in order to modify the hypothesis space, and to speed-up learning and processing times. We study two kinds of filters that are known to be computationally efficient in feed-forward processing: fused convolution/sub-sampling filters, and separable filters. We compare the complexity of the back-propagation algorithm on ConvNets based on these different kinds of filters. We show that using these filters allows to reach the same level of recognition performance as with classical ConvNets for handwritten digit recognition, up to 3.3 times faster.
* [[Learning Separable Filters](https://ieeexplore.ieee.org/document/6619199)] <!-- printed -->
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6619199)]
    * Title: Learning Separable Filters
    * Year: 03 October `2013`
    * Author: Roberto Rigamonti
    * Abstract: Learning filters to produce sparse image representations in terms of over complete dictionaries has emerged as a powerful way to create image features for many different purposes. Unfortunately, these filters are usually both numerous and non-separable, making their use computationally expensive. In this paper, we show that such filters can be computed as linear combinations of a smaller number of separable ones, thus greatly reducing the computational complexity at no cost in terms of performance. This makes filter learning approaches practical even for large images or 3D volumes, and we show that we significantly outperform state-of-the-art methods on the linear structure extraction task, in terms of both accuracy and speed. Moreover, our approach is general and can be used on generic filter banks to reduce the complexity of the convolutions.
    * Comments:
        * > [24] demonstrates that it is possible to relax the rank-1 constraint and essentially rewrite fi as a linear combination of 1D filters.
* [[Training CNNs with Low-Rank Filters for Efficient Image Classification](https://arxiv.org/abs/1511.06744)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1511.06744.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06744/)]
    * Title: Training CNNs with Low-Rank Filters for Efficient Image Classification
    * Year: 20 Nov `2015`
    * Authors: Yani Ioannou, Duncan Robertson, Jamie Shotton, Roberto Cipolla, Antonio Criminisi
    * Institutions: [University of Cambridge], [Microsoft Research]
    * Abstract: We propose a new method for creating computationally efficient convolutional neural networks (CNNs) by using low-rank representations of convolutional filters. Rather than approximating filters in previously-trained networks with more efficient versions, we learn a set of small basis filters from scratch; during training, the network learns to combine these basis filters into more complex filters that are discriminative for image classification. To train such networks, a novel weight initialization scheme is used. This allows effective initialization of connection weights in convolutional layers composed of groups of differently-shaped filters. We validate our approach by applying it to several existing CNN architectures and training these networks from scratch using the CIFAR, ILSVRC and MIT Places datasets. Our results show similar or higher accuracy than conventional CNNs with much less compute. Applying our method to an improved version of VGG-11 network using global max-pooling, we achieve comparable validation accuracy using 41% less compute and only 24% of the original VGG-11 model parameters; another variant of our method gives a 1 percentage point increase in accuracy over our improved VGG-11 model, giving a top-5 center-crop validation accuracy of 89.7% while reducing computation by 16% relative to the original VGG-11 model. Applying our method to the GoogLeNet architecture for ILSVRC, we achieved comparable accuracy with 26% less compute and 41% fewer model parameters. Applying our method to a near state-of-the-art network for CIFAR, we achieved comparable accuracy with 46% less compute and 55% fewer parameters.
    * Comments:
        * > Our approach is connected with that of Ioannou et al. [9] who showed that replacing $3 \times 3 \times c$ filters with linear combinations of filters with smaller spatial extent (e.g. $1 \times 3 \times c$, $3 \times 1 \times c$ filters, see Fig. 3) could reduce the model size and computational complexity of state-of-the-art CNNs, while maintaining or even increasing accuracy. However, that work did not address the channel extent of the filters. (Deep Roots, 2016)

> Various authors have suggested approximating learned convolutional filters using tensor decomposition [11, 13, 18]. 

* [[Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1405.3866.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1405.3866/)]
    * Title: Speeding up Convolutional Neural Networks with Low Rank Expansions
    * Year: 15 May `2014`
    * Authors: Max Jaderberg, Andrea Vedaldi, Andrew Zisserman
    * Institutions: [Visual Geometry Group, University of Oxford]
    * Abstract: The focus of this paper is speeding up the evaluation of convolutional neural networks. While delivering impressive results across a range of computer vision and machine learning tasks, these networks are computationally demanding, limiting their deployability. Convolutional layers generally consume the bulk of the processing time, and so in this work we present two simple schemes for drastically speeding up these layers. This is achieved by exploiting cross-channel or filter redundancy to construct a low rank basis of filters that are rank-1 in the spatial domain. Our methods are architecture agnostic, and can be easily applied to existing CPU and GPU convolutional frameworks for tuneable speedup performance. We demonstrate this with a real world network designed for scene text character recognition, showing a possible 2.5x speedup with no loss in accuracy, and 4.5x speedup with less than 1% drop in accuracy, still achieving state-of-the-art on standard benchmarks.
    * Comments:
        * > Jaderberg et al. [11] propose approximating the convolutional filters in a trained network with representations that are low-rank both in the spatial and the channel domains. (Deep Roots, 2016)
        * > Common approaches first learn these filters from data and then find low-rank approximations as a post-processing step [23]. However, this approach requires additional fine tuning and the resulting filters may not be separable. (2017, ERFNet)
* [[Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition](https://arxiv.org/abs/1412.6553)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1412.6553.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.6553/)]
    * Title: Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition
    * Year: 19 Dec `2014`
    * Authors: Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan Oseledets, Victor Lempitsky
    * Abstract: We propose a simple two-step approach for speeding up convolution layers within large convolutional neural networks based on tensor decomposition and discriminative fine-tuning. Given a layer, we use non-linear least squares to compute a low-rank CP-decomposition of the 4D convolution kernel tensor into a sum of a small number of rank-one tensors. At the second step, this decomposition is used to replace the original convolutional layer with a sequence of four convolutional layers with small kernels. After such replacement, the entire network is fine-tuned on the training data using standard backpropagation process. We evaluate this approach on two CNNs and show that it is competitive with previous approaches, leading to higher obtained CPU speedups at the cost of lower accuracy drops for the smaller of the two networks. Thus, for the 36-class character classification CNN, our approach obtains a 8.5x CPU speedup of the whole network with only minor accuracy drop (1% from 91% to 90%). For the standard ImageNet architecture (AlexNet), the approach speeds up the second convolution layer by a factor of 4x at the cost of $1\%$ increase of the overall top-5 classification error.
* [[Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530)]
    [[pdf](https://arxiv.org/pdf/1511.06530.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06530/)]
    * Title: Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications
    * Year: 20 Nov `2015`
    * Authors: Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim Choi, Lu Yang, Dongjun Shin
    * Abstract: Although the latest high-end smartphone has powerful CPU and GPU, running deeper convolutional neural networks (CNNs) for complex tasks such as ImageNet classification on mobile devices is challenging. To deploy deep CNNs on mobile devices, we present a simple and effective scheme to compress the entire CNN, which we call one-shot whole network compression. The proposed scheme consists of three steps: (1) rank selection with variational Bayesian matrix factorization, (2) Tucker decomposition on kernel tensor, and (3) fine-tuning to recover accumulated loss of accuracy, and each step can be easily implemented using publicly available tools. We demonstrate the effectiveness of the proposed scheme by testing the performance of various compressed CNNs (AlexNet, VGGS, GoogLeNet, and VGG-16) on the smartphone. Significant reductions in model size, runtime, and energy consumption are obtained, at the cost of small loss in accuracy. In addition, we address the important implementation level issue on 1?1 convolution, which is a key operation of inception module of GoogLeNet as well as CNNs compressed by our proposed scheme.

### Tensor Decomposition (ShuffleNet V2, 2018)

* [Efficient and accurate approximations of nonlinear convolutional networks]
* [Accelerating very deep convolutional networks for classification and detection]

## Sparsity

### (2018, ESPNetV1) (count=3)

* [[Sparse Decomposition](https://openaccess.thecvf.com/content_cvpr_2015/html/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.html)]
    * Title: Sparse Convolutional Neural Networks
    * Year: `2015`
    * Authors: Baoyuan Liu, Min Wang, Hassan Foroosh, Marshall Tappen, Marianna Pensky
    * Abstract: Deep neural networks have achieved remarkable performance in both image classification and object detection problems, at the cost of a large number of parameters and computational complexity. In this work, we show how to reduce the redundancy in these parameters using a sparse decomposition. Maximum sparsity is obtained by exploiting both inter-channel and intra-channel redundancy, with a fine-tuning step that minimize the recognition loss caused by maximizing sparsity. This procedure zeros out more than 90\% of parameters, with a drop of accuracy that is less than 1\% on the ILSVRC2012 dataset. We also propose an efficient sparse matrix multiplication algorithm on CPU for Sparse Convolutional Neural Networks (SCNN) models. Our CPU implementation demonstrates much higher efficiency than the off-the-shelf sparse matrix libraries, with a significant speedup realized over the original dense network. In addition, we apply the SCNN model to the object detection problem, in conjunction with a cascade model and sparse fully connected layers, to achieve significant speedups.
* [[Structural Sparsity Learning](https://arxiv.org/abs/1608.03665)]
    [[pdf](https://arxiv.org/pdf/1608.03665.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1608.03665/)]
    * Title: Learning Structured Sparsity in Deep Neural Networks
    * Year: 12 Aug `2016`
    * Authors: Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, Hai Li
    * Abstract: High demand for computation resources severely hinders deployment of large-scale Deep Neural Networks (DNN) in resource constrained devices. In this work, we propose a Structured Sparsity Learning (SSL) method to regularize the structures (i.e., filters, channels, filter shapes, and layer depth) of DNNs. SSL can: (1) learn a compact structure from a bigger DNN to reduce computation cost; (2) obtain a hardware-friendly structured sparsity of DNN to efficiently accelerate the DNNs evaluation. Experimental results show that SSL achieves on average 5.1x and 3.1x speedups of convolutional layer computation of AlexNet against CPU and GPU, respectively, with off-the-shelf libraries. These speedups are about twice speedups of non-structured sparsity; (3) regularize the DNN structure to improve classification accuracy. The results show that for CIFAR-10, regularization on layer depth can reduce 20 layers of a Deep Residual Network (ResNet) to 18 layers while improve the accuracy from 91.25% to 92.60%, which is still slightly higher than that of original ResNet with 32 layers. For AlexNet, structure regularization by SSL also reduces the error by around ~1%. Open source code is in this https URL
* [[LCNN](https://arxiv.org/abs/1611.06473)]
    [[pdf](https://arxiv.org/pdf/1611.06473.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1611.06473/)]
    * Title: LCNN: Lookup-based Convolutional Neural Network
    * Year: 20 Nov `2016`
    * Authors: Hessam Bagherinezhad, Mohammad Rastegari, Ali Farhadi
    * Abstract: Porting state of the art deep learning algorithms to resource constrained compute platforms (e.g. VR, AR, wearables) is extremely challenging. We propose a fast, compact, and accurate model for convolutional neural networks that enables efficient learning and inference. We introduce LCNN, a lookup-based convolutional neural network that encodes convolutions by few lookups to a dictionary that is trained to cover the space of weights in CNNs. Training LCNN involves jointly learning a dictionary and a small set of linear combinations. The size of the dictionary naturally traces a spectrum of trade-offs between efficiency and accuracy. Our experimental results on ImageNet challenge show that LCNN can offer 3.2x speedup while achieving 55.1% top-1 accuracy using AlexNet architecture. Our fastest LCNN offers 37.6x speed up over AlexNet while maintaining 44.3% top-1 accuracy. LCNN not only offers dramatic speed ups at inference, but it also enables efficient training. In this paper, we show the benefits of LCNN in few-shot learning and few-iteration learning, two crucial aspects of on-device training of deep learning models.

### (2018, MobileNetV2) (count=1)

* [[The Power of Sparsity in Convolutional Neural Networks](https://arxiv.org/abs/1702.06257)]
    [[pdf](https://arxiv.org/pdf/1702.06257.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1702.06257/)]
    * Title: The Power of Sparsity in Convolutional Neural Networks
    * Year: 21 Feb `2017`
    * Authors: Soravit Changpinyo, Mark Sandler, Andrey Zhmoginov
    * Institusion: [University of Southern California, Los Angeles]
    * Abstract: Deep convolutional networks are well-known for their high computational and memory demands. Given limited resources, how does one design a network that balances its size, training time, and prediction accuracy? A surprisingly effective approach to trade accuracy for size and speed is to simply reduce the number of channels in each convolutional layer by a fixed fraction and retrain the network. In many cases this leads to significantly smaller networks with only minimal changes to accuracy. In this paper, we take a step further by empirically examining a strategy for deactivating connections between filters in convolutional layers in a way that allows us to harvest savings both in run-time and memory for many network architectures. More specifically, we generalize 2D convolution to use a channel-wise sparse connection structure and show that this leads to significantly better results than the baseline approach for large networks including VGG and Inception V3.

### Depth Completion

* [[HMS-Net: Hierarchical Multi-scale Sparsity-invariant Network for Sparse Depth Completion](https://arxiv.org/abs/1808.08685)]
    [[pdf](https://arxiv.org/pdf/1808.08685.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1808.08685/)]
    * Title: HMS-Net: Hierarchical Multi-scale Sparsity-invariant Network for Sparse Depth Completion
    * Year: 27 Aug `2018`
    * Authors: Zixuan Huang, Junming Fan, Shenggan Cheng, Shuai Yi, Xiaogang Wang, Hongsheng Li
    * Abstract: Dense depth cues are important and have wide applications in various computer vision tasks. In autonomous driving, LIDAR sensors are adopted to acquire depth measurements around the vehicle to perceive the surrounding environments. However, depth maps obtained by LIDAR are generally sparse because of its hardware limitation. The task of depth completion attracts increasing attention, which aims at generating a dense depth map from an input sparse depth map. To effectively utilize multi-scale features, we propose three novel sparsity-invariant operations, based on which, a sparsity-invariant multi-scale encoder-decoder network (HMS-Net) for handling sparse inputs and sparse feature maps is also proposed. Additional RGB features could be incorporated to further improve the depth completion performance. Our extensive experiments and component analysis on two public benchmarks, KITTI depth completion benchmark and NYU-depth-v2 dataset, demonstrate the effectiveness of the proposed approach. As of Aug. 12th, 2018, on KITTI depth completion leaderboard, our proposed model without RGB guidance ranks first among all peer-reviewed methods without using RGB information, and our model with RGB guidance ranks second among all RGB-guided methods.
* [[Sparse and Dense Data with CNNs: Depth Completion and Semantic Segmentation](https://arxiv.org/abs/1808.00769)]
    [[pdf](https://arxiv.org/pdf/1808.00769.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1808.00769/)]
    * Title: Sparse and Dense Data with CNNs: Depth Completion and Semantic Segmentation
    * Year: 02 Aug `2018`
    * Authors: Maximilian Jaritz, Raoul de Charette, Emilie Wirbel, Xavier Perrotton, Fawzi Nashashibi
    * Abstract: Convolutional neural networks are designed for dense data, but vision data is often sparse (stereo depth, point clouds, pen stroke, etc.). We present a method to handle sparse depth data with optional dense RGB, and accomplish depth completion and semantic segmentation changing only the last layer. Our proposal efficiently learns sparse features without the need of an additional validity mask. We show how to ensure network robustness to varying input sparsities. Our method even works with densities as low as 0.8% (8 layer lidar), and outperforms all published state-of-the-art on the Kitti depth completion benchmark.
* [[Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500)]
    [[pdf](https://arxiv.org/pdf/1708.06500.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1708.06500/)]
    * Title: Sparsity Invariant CNNs
    * Year: 22 Aug `2017`
    * Authors: Jonas Uhrig, Nick Schneider, Lukas Schneider, Uwe Franke, Thomas Brox, Andreas Geiger
    * Abstract: In this paper, we consider convolutional neural networks operating on sparse inputs with an application to depth upsampling from sparse laser scan data. First, we show that traditional convolutional networks perform poorly when applied to sparse data even when the location of missing data is provided to the network. To overcome this problem, we propose a simple yet effective sparse convolution layer which explicitly considers the location of missing data during the convolution operation. We demonstrate the benefits of the proposed network architecture in synthetic and real experiments with respect to various baseline approaches. Compared to dense baselines, the proposed sparse convolution network generalizes well to novel datasets and is invariant to the level of sparsity in the data. For our evaluation, we derive a novel dataset from the KITTI benchmark, comprising 93k depth annotated RGB images. Our dataset allows for training and evaluating depth upsampling and depth prediction techniques in challenging real-world settings and will be made available upon publication.
* [[Revisiting Sparsity Invariant Convolution: A Network for Image Guided Depth Completion](https://ieeexplore.ieee.org/document/9138427)]
    * Title: Revisiting Sparsity Invariant Convolution: A Network for Image Guided Depth Completion
    * Year: 10 Jul `2020`
    * Authors: Lin Yan; Kai Liu; Evgeny Belyaev
    * Abstract: The limitation of LiDAR (Light Detection And Ranging) sensor causes the general sparsity of produced depth measurement. However, the sparse representation of the world is insufficient for applications such as 3D reconstruction. Thus, depth completion is an important computer vision task in which a synchronized RGB image is commonly available. In this paper, we propose a deep neural network to tackle this image guided depth completion problem. By revisiting the sparsity invariant convolution and revealing how it can be used in a novel approach, we propose three mask aware operations to process, downscale, and fuse sparse features. These operations explicitly consider the observation mask of its corresponding feature map. In addition, the structure of this network follows a novel scheme in which data from image and depth domain are processed by these proposed operations independently. Our proposed model achieves state-of-the-art performance on the KITTI depth completion benchmark. Furthermore, it presents a strong robustness for bearing input sparsity under different densities and patterns.

## Expressive Power (2019, EfficientNetV1) (4)

> (2019, EfficientNetV1) In fact, previous theoretical (Raghu et al., 2017; Lu et al., 2018) and empirical results (Zagoruyko & Komodakis, 2016) both show that there exists certain relationship between network width and depth.

> (2019, EfficientNetV1) Prior studies (Raghu et al., 2017; Lin & Jegelka, 2018; Sharir & Shashua, 2018; Lu et al., 2018) have shown that network depth and width are both important for ConvNets' expressive power.

* [[On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1606.05336.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1606.05336/)]
    * Title: On the Expressive Power of Deep Neural Networks
    * Year: 16 Jun `2016`
    * Authors: Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, Jascha Sohl-Dickstein
    * Abstract: We propose a new approach to the problem of neural network expressivity, which seeks to characterize how structural properties of a neural network family affect the functions it is able to compute. Our approach is based on an interrelated set of measures of expressivity, unified by the novel notion of trajectory length, which measures how the output of a network changes as the input sweeps along a one-dimensional path. Our findings can be summarized as follows: (1) The complexity of the computed function grows exponentially with depth. (2) All weights are not equal: trained networks are more sensitive to their lower (initial) layer weights. (3) Regularizing on trajectory length (trajectory regularization) is a simpler alternative to batch normalization, with the same performance.
* [[The Expressive Power of Neural Networks: A View from the Width](https://arxiv.org/abs/1709.02540)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1709.02540.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1709.02540/)]
    * Title: The Expressive Power of Neural Networks: A View from the Width
    * Year: 08 Sep `2017`
    * Authors: Zhou Lu, Hongming Pu, Feicheng Wang, Zhiqiang Hu, Liwei Wang
    * Abstract: The expressive power of neural networks is important for understanding deep learning. Most existing works consider this problem from the view of the depth of a network. In this paper, we study how width affects the expressiveness of neural networks. Classical results state that depth-bounded (e.g. depth-$2$) networks with suitable activation functions are universal approximators. We show a universal approximation theorem for width-bounded ReLU networks: width-$(n+4)$ ReLU networks, where $n$ is the input dimension, are universal approximators. Moreover, except for a measure zero set, all functions cannot be approximated by width-$n$ ReLU networks, which exhibits a phase transition. Several recent works demonstrate the benefits of depth by proving the depth-efficiency of neural networks. That is, there are classes of deep networks which cannot be realized by any shallow network whose size is no more than an exponential bound. Here we pose the dual question on the width-efficiency of ReLU networks: Are there wide networks that cannot be realized by narrow networks whose size is not substantially larger? We show that there exist classes of wide networks which cannot be realized by any narrow network whose depth is no more than a polynomial bound. On the other hand, we demonstrate by extensive experiments that narrow networks whose size exceed the polynomial bound by a constant factor can approximate wide and shallow network with high accuracy. Our results provide more comprehensive evidence that depth is more effective than width for the expressiveness of ReLU networks.
* [[On the Expressive Power of Overlapping Architectures of Deep Learning](https://arxiv.org/abs/1703.02065)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1703.02065.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.02065/)]
    * Title: On the Expressive Power of Overlapping Architectures of Deep Learning
    * Year: 06 Mar `2017`
    * Authors: Or Sharir, Amnon Shashua
    * Abstract: Expressive efficiency refers to the relation between two architectures A and B, whereby any function realized by B could be replicated by A, but there exists functions realized by A, which cannot be replicated by B unless its size grows significantly larger. For example, it is known that deep networks are exponentially efficient with respect to shallow networks, in the sense that a shallow network must grow exponentially large in order to approximate the functions represented by a deep network of polynomial size. In this work, we extend the study of expressive efficiency to the attribute of network connectivity and in particular to the effect of "overlaps" in the convolutional process, i.e., when the stride of the convolution is smaller than its filter size (receptive field). To theoretically analyze this aspect of network's design, we focus on a well-established surrogate for ConvNets called Convolutional Arithmetic Circuits (ConvACs), and then demonstrate empirically that our results hold for standard ConvNets as well. Specifically, our analysis shows that having overlapping local receptive fields, and more broadly denser connectivity, results in an exponential increase in the expressive capacity of neural networks. Moreover, while denser connectivity can increase the expressive capacity, we show that the most common types of modern architectures already exhibit exponential increase in expressivity, without relying on fully-connected layers.
* [[ResNet with one-neuron hidden layers is a Universal Approximator](https://arxiv.org/abs/1806.10909)]
    [[pdf](https://arxiv.org/pdf/1806.10909.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1806.10909/)]
    * Title: ResNet with one-neuron hidden layers is a Universal Approximator
    * Year: 28 Jun `2018`
    * Authors: Hongzhou Lin, Stefanie Jegelka
    * Abstract: We demonstrate that a very deep ResNet with stacked modules with one neuron per hidden layer and ReLU activation functions can uniformly approximate any Lebesgue integrable function in $d$ dimensions, i.e. $\ell_1(\mathbb{R}^d)$. Because of the identity mapping inherent to ResNets, our network has alternating layers of dimension one and $d$. This stands in sharp contrast to fully connected networks, which are not universal approximators if their width is the input dimension $d$ [Lu et al, 2017; Hanin and Sellke, 2017]. Hence, our result implies an increase in representational power for narrow deep networks by the ResNet architecture.

## Reparameterization

* [[RepLKNet](https://arxiv.org/abs/2203.06717)]
    [[pdf](https://arxiv.org/pdf/2203.06717.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2203.06717/)]
    * Title: Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs
    * Year: 13 Mar `2022`
    * Authors: Xiaohan Ding, Xiangyu Zhang, Yizhuang Zhou, Jungong Han, Guiguang Ding, Jian Sun
    * Abstract: We revisit large kernel design in modern convolutional neural networks (CNNs). Inspired by recent advances in vision transformers (ViTs), in this paper, we demonstrate that using a few large convolutional kernels instead of a stack of small kernels could be a more powerful paradigm. We suggested five guidelines, e.g., applying re-parameterized large depth-wise convolutions, to design efficient high-performance large-kernel CNNs. Following the guidelines, we propose RepLKNet, a pure CNN architecture whose kernel size is as large as 31x31, in contrast to commonly used 3x3. RepLKNet greatly closes the performance gap between CNNs and ViTs, e.g., achieving comparable or superior results than Swin Transformer on ImageNet and a few typical downstream tasks, with lower latency. RepLKNet also shows nice scalability to big data and large models, obtaining 87.8% top-1 accuracy on ImageNet and 56.0% mIoU on ADE20K, which is very competitive among the state-of-the-arts with similar model sizes. Our study further reveals that, in contrast to small-kernel CNNs, large-kernel CNNs have much larger effective receptive fields and higher shape bias rather than texture bias. Code & models at this https URL.
* [[RepMLP](https://arxiv.org/abs/2105.01883)]
    [[pdf](https://arxiv.org/pdf/2105.01883.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2105.01883/)]
    * Title: RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition
    * Year: 05 May `2021`
    * Authors: Xiaohan Ding, Chunlong Xia, Xiangyu Zhang, Xiaojie Chu, Jungong Han, Guiguang Ding
    * Abstract: We propose RepMLP, a multi-layer-perceptron-style neural network building block for image recognition, which is composed of a series of fully-connected (FC) layers. Compared to convolutional layers, FC layers are more efficient, better at modeling the long-range dependencies and positional patterns, but worse at capturing the local structures, hence usually less favored for image recognition. We propose a structural re-parameterization technique that adds local prior into an FC to make it powerful for image recognition. Specifically, we construct convolutional layers inside a RepMLP during training and merge them into the FC for inference. On CIFAR, a simple pure-MLP model shows performance very close to CNN. By inserting RepMLP in traditional CNN, we improve ResNets by 1.8% accuracy on ImageNet, 2.9% for face recognition, and 2.3% mIoU on Cityscapes with lower FLOPs. Our intriguing findings highlight that combining the global representational capacity and positional perception of FC with the local prior of convolution can improve the performance of neural network with faster speed on both the tasks with translation invariance (e.g., semantic segmentation) and those with aligned images and positional patterns (e.g., face recognition). The code and models are available at this https URL.
* [[RepVGG](https://arxiv.org/abs/2101.03697)]
    [[pdf](https://arxiv.org/pdf/2101.03697.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2101.03697/)]
    * Title: RepVGG: Making VGG-style ConvNets Great Again
    * Year: 11 Jan `2021`
    * Authors: Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun
    * Abstract: We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet. The code and trained models are available at this https URL.

## Regularization Techniques

* [[Dropout](https://arxiv.org/abs/1207.0580)]
    [[pdf](https://arxiv.org/pdf/1207.0580.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1207.0580/)]
    * Title: Improving neural networks by preventing co-adaptation of feature detectors
    * Year: 03 Jul `2012`
    * Authors: Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov
    * Abstract: When a large feedforward neural network is trained on a small training set, it typically performs poorly on held-out test data. This "overfitting" is greatly reduced by randomly omitting half of the feature detectors on each training case. This prevents complex co-adaptations in which a feature detector is only helpful in the context of several other specific feature detectors. Instead, each neuron learns to detect a feature that is generally helpful for producing the correct answer given the combinatorially large variety of internal contexts in which it must operate. Random "dropout" gives big improvements on many benchmark tasks and sets new records for speech and object recognition.
* [Dropout](https://dl.acm.org/doi/10.5555/2627435.2670313)
    * Title: Dropout: a simple way to prevent neural networks from overfitting
    * Year: 01 Jan `2014`
    * Authors: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
    * Abstract: Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. During training, dropout samples from an exponential number of different "thinned" networks. At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods. We show that dropout improves the performance of neural networks on supervised learning tasks in vision, speech recognition, document classification and computational biology, obtaining state-of-the-art results on many benchmark data sets.
    * Comments:
        * > Hinton et al. [5] introduced dropout for regularization of deep networks. When training a network layer with dropout, a random subset of neurons is excluded from both the forward and backward pass for each mini-batch. (Deep Roots, 2016)
* [DropConnect](https://proceedings.mlr.press/v28/wan13.html)
    * Title: Regularization of Neural Networks using DropConnect
    * Year: `2013`
    * Author: Li Wan
    * Abstract: We introduce DropConnect, a generalization of DropOut, for regularizing large fully-connected layers within neural networks. When training with Dropout, a randomly selected subset of activations are set to zero within each layer. DropConnect instead sets a randomly selected subset of weights within the network to zero. Each unit thus receives input from a random subset of units in the previous layer. We derive a bound on the generalization performance of both Dropout and DropConnect. We then evaluate DropConnect on a range of datasets, comparing to Dropout, and show state-of-the-art results on several image recognition benchmarks can be obtained by aggregating multiple DropConnect-trained models.
* [[Maxout](https://arxiv.org/abs/1302.4389)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1302.4389.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1302.4389/)]
    * Title: Maxout Networks
    * Year: 18 Feb `2013`
    * Authors: Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, Yoshua Bengio
    * Abstract: We consider the problem of designing models to leverage a recently introduced approximate model averaging technique called dropout. We define a simple new model called maxout (so named because its output is the max of a set of inputs, and because it is a natural companion to dropout) designed to both facilitate optimization by dropout and improve the accuracy of dropout's fast approximate model averaging technique. We empirically verify that the model successfully accomplishes both of these tasks. We use maxout and dropout to demonstrate state of the art classification performance on four benchmark datasets: MNIST, CIFAR-10, CIFAR-100, and SVHN.
* [Stochastic Pooling](https://arxiv.org/abs/1301.3557)
    [[pdf](https://arxiv.org/pdf/1301.3557.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1301.3557/)]
    * Title: Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
    * Year: 16 Jan `2013`
    * Authors: Matthew D. Zeiler, Rob Fergus
    * Abstract: We introduce a simple and effective method for regularizing large convolutional neural networks. We replace the conventional deterministic pooling operations with a stochastic procedure, randomly picking the activation within each pooling region according to a multinomial distribution, given by the activities within the pooling region. The approach is hyper-parameter free and can be combined with other regularization approaches, such as dropout and data augmentation. We achieve state-of-the-art performance on four image datasets, relative to other approaches that do not utilize data augmentation.
* [DropIn](https://arxiv.org/abs/1511.06951)
    [[pdf](https://arxiv.org/pdf/1511.06951.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06951/)]
    * Title: Gradual DropIn of Layers to Train Very Deep Neural Networks
    * Year: 22 Nov `2015`
    * Authors: Leslie N. Smith, Emily M. Hand, Timothy Doster
    * Abstract: We introduce the concept of dynamically growing a neural network during training. In particular, an untrainable deep network starts as a trainable shallow network and newly added layers are slowly, organically added during training, thereby increasing the network's depth. This is accomplished by a new layer, which we call DropIn. The DropIn layer starts by passing the output from a previous layer (effectively skipping over the newly added layers), then increasingly including units from the new layers for both feedforward and backpropagation. We show that deep networks, which are untrainable with conventional methods, will converge with DropIn layers interspersed in the architecture. In addition, we demonstrate that DropIn provides regularization during training in an analogous way as dropout. Experiments are described with the MNIST dataset and various expanded LeNet architectures, CIFAR-10 dataset with its architecture expanded from 3 to 11 layers, and on the ImageNet dataset with the AlexNet architecture expanded to 13 layers and the VGG 16-layer architecture.

## Data Augmentation

* [[Mixup](https://arxiv.org/abs/1710.09412)]
    [[pdf](https://arxiv.org/pdf/1710.09412.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1710.09412/)]
    * Title: mixup: Beyond Empirical Risk Minimization
    * Year: 25 Oct `2017`
    * Authors: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    * Abstract: Large deep neural networks are powerful, but exhibit undesirable behaviors such as memorization and sensitivity to adversarial examples. In this work, we propose mixup, a simple learning principle to alleviate these issues. In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularizes the neural network to favor simple linear behavior in-between training examples. Our experiments on the ImageNet-2012, CIFAR-10, CIFAR-100, Google commands and UCI datasets show that mixup improves the generalization of state-of-the-art neural network architectures. We also find that mixup reduces the memorization of corrupt labels, increases the robustness to adversarial examples, and stabilizes the training of generative adversarial networks.

## Progressive Learning

* [[FixRes](https://arxiv.org/abs/1906.06423)]
    [[pdf](https://arxiv.org/pdf/1906.06423.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1906.06423/)]
    * Title: Fixing the train-test resolution discrepancy
    * Year: 14 Jun `2019`
    * Authors: Hugo Touvron, Andrea Vedaldi, Matthijs Douze, Hervé Jégou
    * Abstract: Data-augmentation is key to the training of neural networks for image classification. This paper first shows that existing augmentations induce a significant discrepancy between the typical size of the objects seen by the classifier at train and test time. We experimentally validate that, for a target test resolution, using a lower train resolution offers better classification at test time. We then propose a simple yet effective and efficient strategy to optimize the classifier performance when the train and test resolutions differ. It involves only a computationally cheap fine-tuning of the network at the test resolution. This enables training strong classifiers using small training images. For instance, we obtain 77.1% top-1 accuracy on ImageNet with a ResNet-50 trained on 128x128 images, and 79.8% with one trained on 224x224 image. In addition, if we use extra training data we get 82.5% with the ResNet-50 train with 224x224 images. Conversely, when training a ResNeXt-101 32x48d pre-trained in weakly-supervised fashion on 940 million public images at resolution 224x224 and further optimizing for test resolution 320x320, we obtain a test top-1 accuracy of 86.4% (top-5: 98.0%) (single-crop). To the best of our knowledge this is the highest ImageNet single-crop, top-1 and top-5 accuracy to date.
* [[Mix & Match](https://arxiv.org/abs/1908.08986)]
    [[pdf](https://arxiv.org/pdf/1908.08986.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1908.08986/)]
    * Title: Mix & Match: training convnets with mixed image sizes for improved accuracy, speed and scale resiliency
    * Year: 12 Aug `2019`
    * Authors: Elad Hoffer, Berry Weinstein, Itay Hubara, Tal Ben-Nun, Torsten Hoefler, Daniel Soudry
    * Abstract: Convolutional neural networks (CNNs) are commonly trained using a fixed spatial image size predetermined for a given model. Although trained on images of aspecific size, it is well established that CNNs can be used to evaluate a wide range of image sizes at test time, by adjusting the size of intermediate feature maps. In this work, we describe and evaluate a novel mixed-size training regime that mixes several image sizes at training time. We demonstrate that models trained using our method are more resilient to image size changes and generalize well even on small images. This allows faster inference by using smaller images attest time. For instance, we receive a 76.43% top-1 accuracy using ResNet50 with an image size of 160, which matches the accuracy of the baseline model with 2x fewer computations. Furthermore, for a given image size used at test time, we show this method can be exploited either to accelerate training or the final test accuracy. For example, we are able to reach a 79.27% accuracy with a model evaluated at a 288 spatial size for a relative improvement of 14% over the baseline.
    * Comments:
        * > Another closely related work is Mix&Match (Hoffer et al., 2019), which randomly sample different image size for each batch. (EfficientNetV2, 2021)

## Unsupervised Learning

* [[Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/abs/1406.6909)]
    [[pdf](https://arxiv.org/pdf/1406.6909.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1406.6909/)]
    * Title: Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks
    * Year: 26 Jun `2014`
    * Authors: Alexey Dosovitskiy, Philipp Fischer, Jost Tobias Springenberg, Martin Riedmiller, Thomas Brox
    * Abstract: Deep convolutional networks have proven to be very successful in learning task specific features that allow for unprecedented performance on various computer vision tasks. Training of such networks follows mostly the supervised learning paradigm, where sufficiently many input-output pairs are required for training. Acquisition of large training sets is one of the key challenges, when approaching a new task. In this paper, we aim for generic feature learning and present an approach for training a convolutional network using only unlabeled data. To this end, we train the network to discriminate between a set of surrogate classes. Each surrogate class is formed by applying a variety of transformations to a randomly sampled 'seed' image patch. In contrast to supervised network training, the resulting feature representation is not class specific. It rather provides robustness to the transformations that have been applied during training. This generic feature representation allows for classification results that outperform the state of the art for unsupervised learning on several popular datasets (STL-10, CIFAR-10, Caltech-101, Caltech-256). While such generic features cannot compete with class specific features from supervised training on a classification task, we show that they are advantageous on geometric matching problems, where they also outperform the SIFT descriptor.

## Connectivity Learning (2018, MobileNetV2) (2)

* [[Connectivity Learning in Multi-Branch Networks](https://arxiv.org/abs/1709.09582)]
    [[pdf](https://arxiv.org/pdf/1709.09582.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1709.09582/)]
    * Title: Connectivity Learning in Multi-Branch Networks
    * Year: 27 Sep `2017`
    * Authors: Karim Ahmed, Lorenzo Torresani
    * Institutions: [Department of Computer Science, Dartmouth College]
    * Abstract: While much of the work in the design of convolutional networks over the last five years has revolved around the empirical investigation of the importance of depth, filter sizes, and number of feature channels, recent studies have shown that branching, i.e., splitting the computation along parallel but distinct threads and then aggregating their outputs, represents a new promising dimension for significant improvements in performance. To combat the complexity of design choices in multi-branch architectures, prior work has adopted simple strategies, such as a fixed branching factor, the same input being fed to all parallel branches, and an additive combination of the outputs produced by all branches at aggregation points. In this work we remove these predefined choices and propose an algorithm to learn the connections between branches in the network. Instead of being chosen a priori by the human designer, the multi-branch connectivity is learned simultaneously with the weights of the network by optimizing a single loss function defined with respect to the end task. We demonstrate our approach on the problem of multi-class image classification using three different datasets where it yields consistently higher accuracy compared to the state-of-the-art "ResNeXt" multi-branch network given the same learning capacity.
* [[Learning Time/Memory-Efficient Deep Architectures with Budgeted Super Networks](https://arxiv.org/abs/1706.00046)]
    [[pdf](https://arxiv.org/pdf/1706.00046.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1706.00046/)]
    * Title: Learning Time/Memory-Efficient Deep Architectures with Budgeted Super Networks
    * Year: 31 May `2017`
    * Authors: Tom Veniat, Ludovic Denoyer
    * Institutions: [Sorbonne Universités]
    * Abstract: We propose to focus on the problem of discovering neural network architectures efficient in terms of both prediction quality and cost. For instance, our approach is able to solve the following tasks: learn a neural network able to predict well in less than 100 milliseconds or learn an efficient model that fits in a 50 Mb memory. Our contribution is a novel family of models called Budgeted Super Networks (BSN). They are learned using gradient descent techniques applied on a budgeted learning objective function which integrates a maximum authorized cost, while making no assumption on the nature of this cost. We present a set of experiments on computer vision problems and analyze the ability of our technique to deal with three different costs: the computation cost, the memory consumption cost and a distributed computation cost. We particularly show that our model can discover neural network architectures that have a better accuracy than the ResNet and Convolutional Neural Fabrics architectures on CIFAR-10 and CIFAR-100, at a lower cost.

## Unclassified (listed in to be read order)

* [[HRNet](https://arxiv.org/abs/1908.07919)]
    [[pdf](https://arxiv.org/pdf/1908.07919.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1908.07919/)]
    * Title: Deep High-Resolution Representation Learning for Visual Recognition
    * Year: 20 Aug `2019`
    * Authors: Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao
    * Abstract: High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions \emph{in series} (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams \emph{in parallel}; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. We show the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems. All the codes are available at~{\url{this https URL}}.
* [DeCAF](https://arxiv.org/abs/1310.1531)
    [[pdf](https://arxiv.org/pdf/1310.1531.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1310.1531/)]
    * Title: DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition
    * Year: 06 Oct `2013`
    * Authors: Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, Trevor Darrell
    * Abstract: We evaluate whether features extracted from the activation of a deep convolutional network trained in a fully supervised fashion on a large, fixed set of object recognition tasks can be re-purposed to novel generic tasks. Our generic tasks may differ significantly from the originally trained tasks and there may be insufficient labeled or unlabeled data to conventionally train or adapt a deep architecture to the new tasks. We investigate and visualize the semantic clustering of deep convolutional features with respect to a variety of such tasks, including scene recognition, domain adaptation, and fine-grained recognition challenges. We compare the efficacy of relying on various network levels to define a fixed feature, and report novel results that significantly outperform the state-of-the-art on several important vision challenges. We are releasing DeCAF, an open-source implementation of these deep convolutional activation features, along with all associated network parameters to enable vision researchers to be able to conduct experimentation with deep representations across a range of visual concept learning paradigms.
* [[Understanding Deep Architectures using a Recursive Convolutional Network](https://arxiv.org/abs/1312.1847)]
    [[pdf](https://arxiv.org/pdf/1312.1847.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1312.1847/)]
    * Title: Understanding Deep Architectures using a Recursive Convolutional Network
    * Year: 06 Dec `2013`
    * Authors: David Eigen, Jason Rolfe, Rob Fergus, Yann LeCun
    * Abstract: A key challenge in designing convolutional network models is sizing them appropriately. Many factors are involved in these decisions, including number of layers, feature maps, kernel sizes, etc. Complicating this further is the fact that each of these influence not only the numbers and dimensions of the activation units, but also the total number of parameters. In this paper we focus on assessing the independent contributions of three of these linked variables: The numbers of layers, feature maps, and parameters. To accomplish this, we employ a recursive convolutional network whose weights are tied between layers; this allows us to vary each of the three factors in a controlled setting. We find that while increasing the numbers of layers and parameters each have clear benefit, the number of feature maps (and hence dimensionality of the representation) appears ancillary, and finds most of its benefit through the introduction of more weights. Our results (i) empirically confirm the notion that adding layers alone increases computational power, within the context of convolutional layers, and (ii) suggest that precise sizing of convolutional feature map dimensions is itself of little concern; more attention should be paid to the number of parameters in these layers instead.
* [[On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214)]
    [[pdf](https://arxiv.org/pdf/1905.13214.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1905.13214/)]
    * Title: On Network Design Spaces for Visual Recognition
    * Year: 30 May `2019`
    * Authors: Ilija Radosavovic, Justin Johnson, Saining Xie, Wan-Yen Lo, Piotr Dollár
    * Abstract: Over the past several years progress in designing better neural network architectures for visual recognition has been substantial. To help sustain this rate of progress, in this work we propose to reexamine the methodology for comparing network architectures. In particular, we introduce a new comparison paradigm of distribution estimates, in which network design spaces are compared by applying statistical techniques to populations of sampled models, while controlling for confounding factors like network complexity. Compared to current methodologies of comparing point and curve estimates of model families, distribution estimates paint a more complete picture of the entire design landscape. As a case study, we examine design spaces used in neural architecture search (NAS). We find significant statistical differences between recent NAS design space variants that have been largely overlooked. Furthermore, our analysis reveals that the design spaces for standard model families like ResNeXt can be comparable to the more complex ones used in recent NAS work. We hope these insights into distribution analysis will enable more robust progress toward discovering better networks for visual recognition.
* [Flexible, high performance convolutional neural networks for image classification](https://dl.acm.org/doi/10.5555/2283516.2283603)
    * Title: Flexible, high performance convolutional neural networks for image classification
    * Year: 16 July `2011`
    * Authors: Dan C. Cireşan, Ueli Meier, Jonathan Masci, Luca M. Gambardella, Jürgen Schmidhuber
    * Abstract: We present a fast, fully parameterizable GPU implementation of Convolutional Neural Network variants. Our feature extractors are neither carefully designed nor pre-wired, but rather learned in a supervised way. Our deep hierarchical architectures achieve the best published results on benchmarks for object classification (NORB, CIFAR10) and handwritten digit recognition (MNIST), with error rates of 2.53%, 19.51%, 0.35%, respectively. Deep nets trained by simple back-propagation perform better than more shallow ones. Learning is surprisingly rapid. NORB is completely trained within five epochs. Test error rates on MNIST drop to 2.42%, 0.97% and 0.48% after 1, 3 and 17 epochs, respectively.
    * Comments:
        * > Within just a few years, the top-5 image classification accuracy on the 1000-class ImageNet dataset has increased from ~84% [1] to ~95% [2, 3] using deeper networks with rather small receptive fields [4, 5]. (Training Very Deep Networks, 2015)
* [[FitNets](https://arxiv.org/abs/1412.6550)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1412.6550.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.6550/)]
    * Title: FitNets: Hints for Thin Deep Nets
    * Year: 19 Dec `2014`
    * Authors: Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio
    * Abstract: While depth tends to improve network performances, it also makes gradient-based training more difficult since deeper networks tend to be more non-linear. The recently proposed knowledge distillation approach is aimed at obtaining small and fast-to-execute models, and it has shown that a student network could imitate the soft output of a larger teacher network or ensemble of networks. In this paper, we extend this idea to allow the training of a student that is deeper and thinner than the teacher, using not only the outputs but also the intermediate representations learned by the teacher as hints to improve the training process and final performance of the student. Because the student intermediate hidden layer will generally be smaller than the teacher's intermediate hidden layer, additional parameters are introduced to map the student hidden layer to the prediction of the teacher hidden layer. This allows one to train deeper students that can generalize better or run faster, a trade-off that is controlled by the chosen student capacity. For example, on CIFAR-10, a deep student network with almost 10.4 times less parameters outperforms a larger, state-of-the-art teacher network.
    * Comments:
        * > A related recent technique is based on using soft targets from a shallow teacher network to aid in training deeper student networks in multiple stages [25], similar to the neural history compressor for sequences, where a slowly ticking teacher recurrent net is "distilled" into a quickly ticking student recurrent net by forcing the latter to predict the hidden units of the former [26]. (Training Very Deep Networks, 2015)
* [[RegNet](https://arxiv.org/abs/2003.13678)]
    [[pdf](https://arxiv.org/pdf/2003.13678.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2003.13678/)]
    * Title: Designing Network Design Spaces
    * Year: 30 Mar `2020`
    * Authors: Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár
    * Abstract: In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs.
    * Comments:
        * > RegNet (Radosavovic et al., 2020), ResNeSt (Zhang et al., 2020), TResNet (Ridnik et al., 2020), and EfficientNet-X (Li et al., 2021) focus on GPU and/or TPU inference speed. (EfficientNetV2, 2021)
* [[Provable Bounds for Learning Some Deep Representations](https://arxiv.org/abs/1310.6343)]
    [[pdf](https://arxiv.org/pdf/1310.6343.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1310.6343/)]
    * Title: Provable Bounds for Learning Some Deep Representations
    * Year: 23 Oct `2013`
    * Authors: Sanjeev Arora, Aditya Bhaskara, Rong Ge, Tengyu Ma
    * Institutions: [Princeton University, Computer Science Department and Center for Computational Intractability], [Google Research NYC], [Microsoft Research, New England]
    * Abstract: We give algorithms with provable guarantees that learn a class of deep nets in the generative model view popularized by Hinton and others. Our generative model is an $n$ node multilayer neural net that has degree at most $n^{\gamma}$ for some $\gamma <1$ and each edge has a random edge weight in $[-1,1]$. Our algorithm learns {\em almost all} networks in this class with polynomial running time. The sample complexity is quadratic or cubic depending upon the details of the model. The algorithm uses layerwise learning. It is based upon a novel idea of observing correlations among features and using these to infer the underlying edge structure via a global graph recovery procedure. The analysis of the algorithm reveals interesting structure of neural networks with random edge weights.
* [[Exploit the potential of Multi-column architecture for Crowd Counting](https://arxiv.org/abs/2007.05779)]
    [[pdf](https://arxiv.org/pdf/2007.05779.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2007.05779/)]
    * Title: Exploit the potential of Multi-column architecture for Crowd Counting
    * Year: 11 Jul `2020`
    * Authors: Junhao Cheng, Zhuojun Chen, XinYu Zhang, Yizhou Li, Xiaoyuan Jing
    * Abstract: Crowd counting is an important yet challenging task in computer vision due to serious occlusions, complex background and large scale variations, etc. Multi-column architecture is widely adopted to overcome these challenges, yielding state-of-the-art performance in many public benchmarks. However, there still are two issues in such design: scale limitation and feature similarity. Further performance improvements are thus restricted. In this paper, we propose a novel crowd counting framework called Pyramid Scale Network (PSNet) to explicitly address these issues. Specifically, for scale limitation, we adopt three Pyramid Scale Modules (PSM) to efficiently capture multi-scale features, which integrate a message passing mechanism and an attention mechanism into multi-column architecture. Moreover, for feature similarity, a novel loss function named Multi-column variance loss is introduced to make the features learned by each column in PSM appropriately different from each other. To the best of our knowledge, PSNet is the first work to explicitly address scale limitation and feature similarity in multi-column design. Extensive experiments on five benchmark datasets demonstrate the effectiveness of the proposed innovations as well as the superior performance over the state-of-the-art. Our code is publicly available at: this https URL
* [[An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)]
    [[pdf](https://arxiv.org/pdf/1807.03247.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1807.03247/)]
    * Title: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    * Year: 09 Jul `2018`
    * Authors: Rosanne Liu, Joel Lehman, Piero Molino, Felipe Petroski Such, Eric Frank, Alex Sergeev, Jason Yosinski
    * Abstract: Few ideas have enjoyed as large an impact on deep learning as convolution. For any problem involving pixels or spatial representations, common intuition holds that convolutional neural networks may be appropriate. In this paper we show a striking counterexample to this intuition via the seemingly trivial coordinate transform problem, which simply requires learning a mapping between coordinates in (x,y) Cartesian space and one-hot pixel space. Although convolutional networks would seem appropriate for this task, we show that they fail spectacularly. We demonstrate and carefully analyze the failure first on a toy problem, at which point a simple fix becomes obvious. We call this solution CoordConv, which works by giving convolution access to its own input coordinates through the use of extra coordinate channels. Without sacrificing the computational and parametric efficiency of ordinary convolution, CoordConv allows networks to learn either complete translation invariance or varying degrees of translation dependence, as required by the end task. CoordConv solves the coordinate transform problem with perfect generalization and 150 times faster with 10--100 times fewer parameters than convolution. This stark contrast raises the question: to what extent has this inability of convolution persisted insidiously inside other tasks, subtly hampering performance from within? A complete answer to this question will require further investigation, but we show preliminary evidence that swapping convolution for CoordConv can improve models on a diverse set of tasks. Using CoordConv in a GAN produced less mode collapse as the transform between high-level spatial latents and pixels becomes easier to learn. A Faster R-CNN detection model trained on MNIST showed 24% better IOU when using CoordConv, and in the RL domain agents playing Atari games benefit significantly from the use of CoordConv layers.
    * Comments:
        * > (2019, AANet) Multiple positional encodings that augment activation maps with explicit spatial information have been proposed to alleviate related issues. In particular, the Image Transformer [32] extends the sinusoidal waves first introduced in the original Transformer [43] to 2 dimensional inputs and CoordConv [29] concatenates positional channels to an activation map.
* [[Do Better ImageNet Models Transfer Better?](https://arxiv.org/abs/1805.08974)]
    [[pdf](https://arxiv.org/pdf/1805.08974.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1805.08974/)]
    * Title: Do Better ImageNet Models Transfer Better?
    * Year: 23 May `2018`
    * Authors: Simon Kornblith, Jonathon Shlens, Quoc V. Le
    * Abstract: Transfer learning is a cornerstone of computer vision, yet little work has been done to evaluate the relationship between architecture and transfer. An implicit hypothesis in modern computer vision research is that models that perform better on ImageNet necessarily perform better on other vision tasks. However, this hypothesis has never been systematically tested. Here, we compare the performance of 16 classification networks on 12 image classification datasets. We find that, when networks are used as fixed feature extractors or fine-tuned, there is a strong correlation between ImageNet accuracy and transfer accuracy ($r = 0.99$ and $0.96$, respectively). In the former setting, we find that this relationship is very sensitive to the way in which networks are trained on ImageNet; many common forms of regularization slightly improve ImageNet accuracy but yield penultimate layer features that are much worse for transfer learning. Additionally, we find that, on two small fine-grained image classification datasets, pretraining on ImageNet provides minimal benefits, indicating the learned features from ImageNet do not transfer well to fine-grained tasks. Together, our results show that ImageNet architectures generalize well across datasets, but ImageNet features are less general than previously suggested.
