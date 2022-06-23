<span style="font-family:monospace">

# Papers in Computer Vision - CNN Architectures

count: 31

* [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
    * Title: ImageNet Classification with Deep Convolutional Neural Networks
    * Year: `2012`
    * Author: Alex Krizhevsky
* [Network In Network](https://arxiv.org/abs/1312.4400)
    * Title: Network In Network
    * Year: 16 Dec `2013`
    * Author: Min Lin
* [VGG](https://arxiv.org/abs/1409.1556)
    * Title: Very Deep Convolutional Networks for Large-Scale Image Recognition
    * Year: 04 Sep `2014`
    * Author: Karen Simonyan
    * Abstract: In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localization and classification tracks respectively. We also show that our representations generalize well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.
* [Flattened Networks](https://arxiv.org/abs/1412.5474)
    * Title: Flattened Convolutional Neural Networks for Feedforward Acceleration
    * Year: 17 Dec `2014`
    * Author: Jonghoon Jin
    * Abstract: We present flattened convolutional neural networks that are designed for fast feedforward execution. The redundancy of the parameters, especially weights of the convolutional filters in convolutional neural networks has been extensively studied and different heuristics have been proposed to construct a low rank basis of the filters after training. In this work, we train flattened networks that consist of consecutive sequence of one-dimensional filters across all directions in 3D space to obtain comparable performance as conventional convolutional networks. We tested flattened model on different datasets and found that the flattened layer can effectively substitute for the 3D filters without loss of accuracy. The flattened convolution pipelines provide around two times speed-up during feedforward pass compared to the baseline model due to the significant reduction of learning parameters. Furthermore, the proposed method does not require efforts in manual tuning or post processing once the model is trained.
* [Highway Networks - Report](https://arxiv.org/abs/1505.00387)
    * Title: Highway Networks
    * Year: 03 May `2015`
    * Author: Rupesh Kumar Srivastava
    * Abstract: There is plenty of theoretical and empirical evidence that depth of neural networks is a crucial ingredient for their success. However, network training becomes more difficult with increasing depth and training of very deep networks remains an open problem. In this extended abstract, we introduce a new architecture designed to ease gradient-based training of very deep networks. We refer to networks with this architecture as highway networks, since they allow unimpeded information flow across several layers on "information highways". The architecture is characterized by the use of gating units which learn to regulate the flow of information through a network. Highway networks with hundreds of layers can be trained directly using stochastic gradient descent and with a variety of activation functions, opening up the possibility of studying extremely deep and efficient architectures.
* [Highway Networks](https://arxiv.org/abs/1507.06228)
    * Title: Training Very Deep Networks
    * Year: 22 Jul `2015`
    * Author: Rupesh Kumar Srivastava
    * Abstract: Theoretical and empirical evidence indicates that the depth of neural networks is crucial for their success. However, training becomes more difficult as depth increases, and training of very deep networks remains an open problem. Here we introduce a new architecture designed to overcome this. Our so-called highway networks allow unimpeded information flow across many layers on information highways. They are inspired by Long Short-Term Memory recurrent networks and use adaptive gating units to regulate the information flow. Even with hundreds of layers, highway networks can be trained directly through simple gradient descent. This enables the study of extremely deep and efficient architectures.
* [SqueezeNet](https://arxiv.org/abs/1602.07360)
    * Title: SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
    * Year: 24 Feb `2016`
    * Author: Forrest N. Iandola
    * Abstract: Recent research on deep neural networks has focused primarily on improving accuracy. For a given accuracy level, it is typically possible to identify multiple DNN architectures that achieve that accuracy level. With equivalent accuracy, smaller DNN architectures offer at least three advantages: (1) Smaller DNNs require less communication across servers during distributed training. (2) Smaller DNNs require less bandwidth to export a new model from the cloud to an autonomous car. (3) Smaller DNNs are more feasible to deploy on FPGAs and other hardware with limited memory. To provide all of these advantages, we propose a small DNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet). The SqueezeNet architecture is available for download here: [this https URL](https://github.com/DeepScale/SqueezeNet).
* [FractalNet](https://arxiv.org/abs/1605.07648)
    * Title: FractalNet: Ultra-Deep Neural Networks without Residuals
    * Year: 24 May `2016`
    * Author: Gustav Larsson
* [DFN](https://arxiv.org/abs/1605.07716)
    * Title: Deeply-Fused Nets
    * Year: 25 May `2016`
    * Author: Jingdong Wang
* [DenseNet](https://arxiv.org/abs/1608.06993)
    * Title: Densely Connected Convolutional Networks
    * Year: 25 Aug `2016`
    * Author: Gao Huang
* [Xception Networks](https://arxiv.org/abs/1610.02357)
    * Title: Xception: Deep Learning with Depthwise Separable Convolutions
    * Year: 07 Oct `2016`
    * Author: François Chollet
    * Abstract: We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.
* [Deformable](https://arxiv.org/abs/1703.06211)
    * Title: Deformable Convolutional Networks
    * Year: 17 Mar 2017
    * Author: Jifeng Dai
    * Abstract: Convolutional neural networks (CNNs) are inherently limited to model geometric transformations due to the fixed geometric structures in its building modules. In this work, we introduce two new modules to enhance the transformation modeling capacity of CNNs, namely, deformable convolution and deformable RoI pooling. Both are based on the idea of augmenting the spatial sampling locations in the modules with additional offsets and learning the offsets from target tasks, without additional supervision. The new modules can readily replace their plain counterparts in existing CNNs and can be easily trained end-to-end by standard back-propagation, giving rise to deformable convolutional networks. Extensive experiments validate the effectiveness of our approach on sophisticated vision tasks of object detection and semantic segmentation. The code would be released.
* [ConvNeXt](https://arxiv.org/abs/2201.03545)
    * Title: A ConvNet for the 2020s
    * Year: 10 Jan `2022`
    * Author: Zhuang Liu

## Residual Networks

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
* [Wide ResNet](https://arxiv.org/abs/1605.07146)
    * Title: Wide Residual Networks
    * Year: 23 May `2016`
    * Author: Sergey Zagoruyko
* [ResNeXt](https://arxiv.org/abs/1611.05431)
    * Title: Aggregated Residual Transformations for Deep Neural Networks
    * Year: 16 Nov `2016`
    * Author: Saining Xie
    * Abstract: We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call "cardinality" (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online.

## Inception Networks

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
* [Inception-v4](https://arxiv.org/abs/1602.07261)
    * Title: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    * Year: 23 Feb `2016`
    * Author: Christian Szegedy
    * Abstract: Very deep convolutional networks have been central to the largest advances in image recognition performance in recent years. One example is the Inception architecture that has been shown to achieve very good performance at relatively low computational cost. Recently, the introduction of residual connections in conjunction with a more traditional architecture has yielded state-of-the-art performance in the 2015 ILSVRC challenge; its performance was similar to the latest generation Inception-v3 network. This raises the question of whether there are any benefit in combining the Inception architecture with residual connections. Here we give clear empirical evidence that training with residual connections accelerates the training of Inception networks significantly. There is also some evidence of residual Inception networks outperforming similarly expensive Inception networks without residual connections by a thin margin. We also present several new streamlined architectures for both residual and non-residual Inception networks. These variations improve the single-frame recognition performance on the ILSVRC 2012 classification task significantly. We further demonstrate how proper activation scaling stabilizes the training of very wide residual Inception networks. With an ensemble of three residual and one Inception-v4, we achieve 3.08 percent top-5 error on the test set of the ImageNet classification (CLS) challenge

## Mobile Networks

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

## Regularization Techniques

* [DropConnect](https://proceedings.mlr.press/v28/wan13.html)
    * Title: Regularization of Neural Networks using DropConnect
    * Year: `2013`
    * Author: Li Wan
    * Abstract: We introduce DropConnect, a generalization of DropOut, for regularizing large fully-connected layers within neural networks. When training with Dropout, a randomly selected subset of activations are set to zero within each layer. DropConnect instead sets a randomly selected subset of weights within the network to zero. Each unit thus receives input from a random subset of units in the previous layer. We derive a bound on the generalization performance of both Dropout and DropConnect. We then evaluate DropConnect on a range of datasets, comparing to Dropout, and show state-of-the-art results on several image recognition benchmarks can be obtained by aggregating multiple DropConnect-trained models.
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

## Knowledge Distillation

* [Distillation](https://arxiv.org/abs/1503.02531)
    * Title: Distilling the Knowledge in a Neural Network
    * Year: 09 Mar `2015`
    * Author: Geoffrey Hinton
    * Abstract: A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.
