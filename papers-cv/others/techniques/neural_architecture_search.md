# [Papers][Vision] Network Architecture Search (NAS) <!-- omit in toc -->

count=13

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Survey](#survey)
- [Architecture Search (2018, MobileNetV2) (count=4)](#architecture-search-2018-mobilenetv2-count4)
- [Reinforcement Learning Based (MobileNetV3, 2019) (5)](#reinforcement-learning-based-mobilenetv3-2019-5)
- [Block-Level Hierarchical Search (MobileNetV3, 2019) (1)](#block-level-hierarchical-search-mobilenetv3-2019-1)
- [Differentiable Architecture Search Frameworks (MobileNetV3, 2019) (3)](#differentiable-architecture-search-frameworks-mobilenetv3-2019-3)
- [Network Simplification Algorithms (MobileNetV3, 2019) (3)](#network-simplification-algorithms-mobilenetv3-2019-3)
- [(2019, EfficientNetV1)](#2019-efficientnetv1)
- [Improving inference efficiency (EfficientNetV2, 2021)](#improving-inference-efficiency-efficientnetv2-2021)
- [Unknown](#unknown)

----------------------------------------------------------------------------------------------------

## Survey

* [Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377)
    * Title: Neural Architecture Search: A Survey
    * Year: 16 Aug `2018`
    * Authors: Thomas Elsken, Jan Hendrik Metzen, Frank Hutter
    * Abstract: Deep Learning has enabled remarkable progress over the last years on a variety of tasks, such as image recognition, speech recognition, and machine translation. One crucial aspect for this progress are novel neural architectures. Currently employed architectures have mostly been developed manually by human experts, which is a time-consuming and error-prone process. Because of this, there is growing interest in automated neural architecture search methods. We provide an overview of existing work in this field of research and categorize them according to three dimensions: search space, search strategy, and performance estimation strategy.

## Architecture Search (2018, MobileNetV2) (count=4)

> (2018, MobileNetV2) Recently, [22, 23, 24, 25], opened up a new direction of bringing optimization methods including genetic algorithms and reinforcement learning to architectural search. However one drawback is that the resulting networks end up very complex.

* [[NASNet](https://arxiv.org/abs/1707.07012)]
    [[pdf](https://arxiv.org/pdf/1707.07012.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1707.07012/)]
    * Title: Learning Transferable Architectures for Scalable Image Recognition
    * Year: 21 Jul `2017`
    * Authors: Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
    * Abstract: Developing neural network image classification models often requires significant architecture engineering. In this paper, we study a method to learn the model architectures directly on the dataset of interest. As this approach is expensive when the dataset is large, we propose to search for an architectural building block on a small dataset and then transfer the block to a larger dataset. The key contribution of this work is the design of a new search space (the "NASNet search space") which enables transferability. In our experiments, we search for the best convolutional layer (or "cell") on the CIFAR-10 dataset and then apply this cell to the ImageNet dataset by stacking together more copies of this cell, each with their own parameters to design a convolutional architecture, named "NASNet architecture". We also introduce a new regularization technique called ScheduledDropPath that significantly improves generalization in the NASNet models. On CIFAR-10 itself, NASNet achieves 2.4% error rate, which is state-of-the-art. On ImageNet, NASNet achieves, among the published works, state-of-the-art accuracy of 82.7% top-1 and 96.2% top-5 on ImageNet. Our model is 1.2% better in top-1 accuracy than the best human-invented architectures while having 9 billion fewer FLOPS - a reduction of 28% in computational demand from the previous state-of-the-art model. When evaluated at different levels of computational cost, accuracies of NASNets exceed those of the state-of-the-art human-designed models. For instance, a small version of NASNet also achieves 74% top-1 accuracy, which is 3.1% better than equivalently-sized, state-of-the-art models for mobile platforms. Finally, the learned features by NASNet used with the Faster-RCNN framework surpass state-of-the-art by 4.0% achieving 43.1% mAP on the COCO dataset.
* [[Genetic CNN](https://arxiv.org/abs/1703.01513)]
    [[pdf](https://arxiv.org/pdf/1703.01513.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.01513/)]
    * Title: Genetic CNN
    * Year: 04 Mar `2017`
    * Authors: Lingxi Xie, Alan Yuille
    * Institutions: [Center for Imaging Science, The Johns Hopkins University]
    * Abstract: The deep Convolutional Neural Network (CNN) is the state-of-the-art solution for large-scale visual recognition. Following basic principles such as increasing the depth and constructing highway connections, researchers have manually designed a lot of fixed network structures and verified their effectiveness. In this paper, we discuss the possibility of learning deep network structures automatically. Note that the number of possible network structures increases exponentially with the number of layers in the network, which inspires us to adopt the genetic algorithm to efficiently traverse this large search space. We first propose an encoding method to represent each network structure in a fixed-length binary string, and initialize the genetic algorithm by generating a set of randomized individuals. In each generation, we define standard genetic operations, e.g., selection, mutation and crossover, to eliminate weak individuals and then generate more competitive ones. The competitiveness of each individual is defined as its recognition accuracy, which is obtained via training the network from scratch and evaluating it on a validation set. We run the genetic process on two small datasets, i.e., MNIST and CIFAR10, demonstrating its ability to evolve and find high-quality structures which are little studied before. These structures are also transferrable to the large-scale ILSVRC2012 dataset.
* [[Large-Scale Evolution of Image Classifiers](https://arxiv.org/abs/1703.01041)]
    [[pdf](https://arxiv.org/pdf/1703.01041.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.01041/)]
    * Title: Large-Scale Evolution of Image Classifiers
    * Year: 03 Mar `2017`
    * Authors: Esteban Real, Sherry Moore, Andrew Selle, Saurabh Saxena, Yutaka Leon Suematsu, Jie Tan, Quoc Le, Alex Kurakin
    * Institutions: [Google Brain], [Google Research]
    * Abstract: Neural networks have proven effective at solving difficult problems but designing their architectures can be challenging, even for image classification problems alone. Our goal is to minimize human participation, so we employ evolutionary algorithms to discover such networks automatically. Despite significant computational requirements, we show that it is now possible to evolve models with accuracies within the range of those published in the last year. Specifically, we employ simple evolutionary techniques at unprecedented scales to discover models for the CIFAR-10 and CIFAR-100 datasets, starting from trivial initial conditions and reaching accuracies of 94.6% (95.6% for ensemble) and 77.0%, respectively. To do this, we use novel and intuitive mutation operators that navigate large search spaces; we stress that no human participation is required once evolution starts and that the output is a fully-trained model. Throughout this work, we place special emphasis on the repeatability of results, the variability in the outcomes and the computational requirements.
* [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)
    * Title: Neural Architecture Search with Reinforcement Learning
    * Year: 05 Nov `2016`
    * Author: Barret Zoph
    * Abstract: Neural networks are powerful and flexible models that work well for many difficult learning tasks in image, speech and natural language understanding. Despite their success, neural networks are still hard to design. In this paper, we use a recurrent network to generate the model descriptions of neural networks and train this RNN with reinforcement learning to maximize the expected accuracy of the generated architectures on a validation set. On the CIFAR-10 dataset, our method, starting from scratch, can design a novel network architecture that rivals the best human-invented architecture in terms of test set accuracy. Our CIFAR-10 model achieves a test error rate of 3.65, which is 0.09 percent better and 1.05x faster than the previous state-of-the-art model that used a similar architectural scheme. On the Penn Treebank dataset, our model can compose a novel recurrent cell that outperforms the widely-used LSTM cell, and other state-of-the-art baselines. Our cell achieves a test set perplexity of 62.4 on the Penn Treebank, which is 3.6 perplexity better than the previous state-of-the-art model. The cell can also be transferred to the character language modeling task on PTB and achieves a state-of-the-art perplexity of 1.214.

## Reinforcement Learning Based (MobileNetV3, 2019) (5)

* Neural Architecture Search with Reinforcement Learning
* Learning Transferable Architectures for Scalable Image Recognition
* [Designing Neural Network Architectures using Reinforcement Learning](https://arxiv.org/abs/1611.02167)
    * Title: Designing Neural Network Architectures using Reinforcement Learning
    * Year: 07 Nov `2016`
    * Author: Bowen Baker
    * Abstract: At present, designing convolutional neural network (CNN) architectures requires both human expertise and labor. New architectures are handcrafted by careful experimentation or modified from a handful of existing networks. We introduce MetaQNN, a meta-modeling algorithm based on reinforcement learning to automatically generate high-performing CNN architectures for a given learning task. The learning agent is trained to sequentially choose CNN layers using Q-learning with an $\epsilon$-greedy exploration strategy and experience replay. The agent explores a large but finite space of possible architectures and iteratively discovers designs with improved performance on the learning task. On image classification benchmarks, the agent-designed networks (consisting of only standard convolution, pooling, and fully-connected layers) beat existing networks designed with the same layer types and are competitive against the state-of-the-art methods that use more complex layer types. We also outperform existing meta-modeling approaches for network design on image classification tasks.
* [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559)
    * Title: Progressive Neural Architecture Search
    * Year: 02 Dec `2017`
    * Author: Chenxi Liu
    * Abstract: We propose a new method for learning the structure of convolutional neural networks (CNNs) that is more efficient than recent state-of-the-art methods based on reinforcement learning and evolutionary algorithms. Our approach uses a sequential model-based optimization (SMBO) strategy, in which we search for structures in order of increasing complexity, while simultaneously learning a surrogate model to guide the search through structure space. Direct comparison under the same search space shows that our method is up to 5 times more efficient than the RL method of Zoph et al. (2018) in terms of number of models evaluated, and 8 times faster in terms of total compute. The structures we discover in this way achieve state of the art classification accuracies on CIFAR-10 and ImageNet.
* [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268)
    * Title: Efficient Neural Architecture Search via Parameter Sharing
    * Year: 09 Feb `2018`
    * Author: Hieu Pham
    * Abstract: We propose Efficient Neural Architecture Search (ENAS), a fast and inexpensive approach for automatic model design. In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. The controller is trained with policy gradient to select a subgraph that maximizes the expected reward on the validation set. Meanwhile the model corresponding to the selected subgraph is trained to minimize a canonical cross entropy loss. Thanks to parameter sharing between child models, ENAS is fast: it delivers strong empirical performances using much fewer GPU-hours than all existing automatic model design approaches, and notably, 1000x less expensive than standard Neural Architecture Search. On the Penn Treebank dataset, ENAS discovers a novel architecture that achieves a test perplexity of 55.8, establishing a new state-of-the-art among all methods without post-training processing. On the CIFAR-10 dataset, ENAS designs novel architectures that achieve a test error of 2.89%, which is on par with NASNet (Zoph et al., 2018), whose test error is 2.65%.

## Block-Level Hierarchical Search (MobileNetV3, 2019) (1)

* [[MnasNet](https://arxiv.org/abs/1807.11626)] <!-- printed -->
    * Title: MnasNet: Platform-Aware Neural Architecture Search for Mobile
    * Year: 31 Jul `2018`
    * Authors: Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le
    * Abstract: Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider. In this paper, we propose an automated mobile neural architecture search (MNAS) approach, which explicitly incorporate model latency into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. Unlike previous work, where latency is considered via another, often inaccurate proxy (e.g., FLOPS), our approach directly measures real-world inference latency by executing the model on mobile phones. To further strike the right balance between flexibility and search space size, we propose a novel factorized hierarchical search space that encourages layer diversity throughout the network. Experimental results show that our approach consistently outperforms state-of-the-art mobile CNN models across multiple vision tasks. On the ImageNet classification task, our MnasNet achieves 75.2% top-1 accuracy with 78ms latency on a Pixel phone, which is 1.8x faster than MobileNetV2 [29] with 0.5% higher accuracy and 2.3x faster than NASNet [36] with 1.2% higher accuracy. Our MnasNet also achieves better mAP quality than MobileNets for COCO object detection. Code is at [this https URL](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet).

## Differentiable Architecture Search Frameworks (MobileNetV3, 2019) (3)

## Network Simplification Algorithms (MobileNetV3, 2019) (3)

## (2019, EfficientNetV1)

* [ProxylessNAS](https://arxiv.org/abs/1812.00332)
    * Title: ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
    * Year: 02 Dec `2018`
    * Authors: Han Cai, Ligeng Zhu, Song Han
    * Abstract: Neural architecture search (NAS) has a great impact by automatically designing effective neural network architectures. However, the prohibitive computational demand of conventional NAS algorithms (e.g. $10^{4}$ GPU hours) makes it difficult to \emph{directly} search the architectures on large-scale tasks (e.g. ImageNet). Differentiable NAS can reduce the cost of GPU hours via a continuous representation of network architecture but suffers from the high GPU memory consumption issue (grow linearly w.r.t. candidate set size). As a result, they need to utilize~\emph{proxy} tasks, such as training on a smaller dataset, or learning with only a few blocks, or training just for a few epochs. These architectures optimized on proxy tasks are not guaranteed to be optimal on the target task. In this paper, we present \emph{ProxylessNAS} that can \emph{directly} learn the architectures for large-scale target tasks and target hardware platforms. We address the high memory consumption issue of differentiable NAS and reduce the computational cost (GPU hours and GPU memory) to the same level of regular training while still allowing a large candidate set. Experiments on CIFAR-10 and ImageNet demonstrate the effectiveness of directness and specialization. On CIFAR-10, our model achieves 2.08\% test error with only 5.7M parameters, better than the previous state-of-the-art architecture AmoebaNet-B, while using 6x fewer parameters. On ImageNet, our model achieves 3.1\% better top-1 accuracy than MobileNetV2, while being 1.2x faster with measured GPU latency. We also apply ProxylessNAS to specialize neural architectures for hardware with direct hardware metrics (e.g. latency) and provide insights for efficient CNN architecture design.
* [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/abs/1802.01548)
    * Title: Regularized Evolution for Image Classifier Architecture Search
    * Year: 05 Feb `2018`
    * Authors: Esteban Real, Alok Aggarwal, Yanping Huang, Quoc V Le
    * Abstract: The effort devoted to hand-crafting neural network image classifiers has motivated the use of architecture search to discover them automatically. Although evolutionary algorithms have been repeatedly applied to neural network topologies, the image classifiers thus discovered have remained inferior to human-crafted ones. Here, we evolve an image classifier---AmoebaNet-A---that surpasses hand-designs for the first time. To do this, we modify the tournament selection evolutionary algorithm by introducing an age property to favor the younger genotypes. Matching size, AmoebaNet-A has comparable accuracy to current state-of-the-art ImageNet models discovered with more complex architecture-search methods. Scaled to larger size, AmoebaNet-A sets a new state-of-the-art 83.9% / 96.6% top-5 ImageNet accuracy. In a controlled comparison against a well known reinforcement learning algorithm, we give evidence that evolution can obtain results faster with the same hardware, especially at the earlier stages of the search. This is relevant when fewer compute resources are available. Evolution is, thus, a simple method to effectively discover high-quality architectures.

## Improving inference efficiency (EfficientNetV2, 2021)

* MnasNet: Platform-Aware Neural Architecture Search for Mobile
* ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
* [FBNet](https://arxiv.org/abs/1812.03443)
    * Title: FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
    * Year: 09 Dec `2018`
    * Authors: Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, Kurt Keutzer
    * Abstract: Designing accurate and efficient ConvNets for mobile devices is challenging because the design space is combinatorially large. Due to this, previous neural architecture search (NAS) methods are computationally expensive. ConvNet architecture optimality depends on factors such as input resolution and target devices. However, existing approaches are too expensive for case-by-case redesigns. Also, previous work focuses primarily on reducing FLOPs, but FLOP count does not always reflect actual latency. To address these, we propose a differentiable neural architecture search (DNAS) framework that uses gradient-based methods to optimize ConvNet architectures, avoiding enumerating and training individual architectures separately as in previous methods. FBNets, a family of models discovered by DNAS surpass state-of-the-art models both designed manually and generated automatically. FBNet-B achieves 74.1% top-1 accuracy on ImageNet with 295M FLOPs and 23.1 ms latency on a Samsung S8 phone, 2.4x smaller and 1.5x faster than MobileNetV2-1.3 with similar accuracy. Despite higher accuracy and lower latency than MnasNet, we estimate FBNet-B's search cost is 420x smaller than MnasNet's, at only 216 GPU-hours. Searched for different resolutions and channel sizes, FBNets achieve 1.5% to 6.4% higher accuracy than MobileNetV2. The smallest FBNet achieves 50.2% accuracy and 2.9 ms latency (345 frames per second) on a Samsung S8. Over a Samsung-optimized FBNet, the iPhone-X-optimized model achieves a 1.4x speedup on an iPhone X.
* Searching for Fast Model Families on Datacenter Accelerators (See CNN.md)

## Unknown

* [[Searching for Efficient Multi-Scale Architectures for Dense Image Prediction](https://arxiv.org/abs/1809.04184)]
    [[pdf](https://arxiv.org/pdf/1809.04184.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1809.04184/)]
    * Title: Searching for Efficient Multi-Scale Architectures for Dense Image Prediction
    * Year: 11 Sep `2018`
    * Authors: Liang-Chieh Chen, Maxwell D. Collins, Yukun Zhu, George Papandreou, Barret Zoph, Florian Schroff, Hartwig Adam, Jonathon Shlens
    * Abstract: The design of neural network architectures is an important component for achieving state-of-the-art performance with machine learning systems across a broad array of tasks. Much work has endeavored to design and build architectures automatically through clever construction of a search space paired with simple learning algorithms. Recent progress has demonstrated that such meta-learning methods may exceed scalable human-invented architectures on image classification tasks. An open question is the degree to which such methods may generalize to new domains. In this work we explore the construction of meta-learning techniques for dense image prediction focused on the tasks of scene parsing, person-part segmentation, and semantic image segmentation. Constructing viable search spaces in this domain is challenging because of the multi-scale representation of visual information and the necessity to operate on high resolution imagery. Based on a survey of techniques in dense image prediction, we construct a recursive search space and demonstrate that even with efficient random search, we can identify architectures that outperform human-invented architectures and achieve state-of-the-art performance on three dense prediction tasks including 82.7\% on Cityscapes (street scene parsing), 71.3\% on PASCAL-Person-Part (person-part segmentation), and 87.9\% on PASCAL VOC 2012 (semantic image segmentation). Additionally, the resulting architecture is more computationally efficient, requiring half the parameters and half the computational cost as previous state of the art systems.
