# [Papers][Vision] 2D Object Detection <!-- omit in toc -->

count=88

## Table of Contents <!-- omit in toc -->

- [Unclassified](#unclassified)
- [Localization](#localization)
- [Sliding-Window Approaches](#sliding-window-approaches)
- [Object Proposal Approaches (DeepMask, 2015)](#object-proposal-approaches-deepmask-2015)
- [Region Proposals (2013, R-CNN)](#region-proposals-2013-r-cnn)
- [Two-Stage Detectors](#two-stage-detectors)
  - [unclassified](#unclassified-1)
  - [R-CNN Series](#r-cnn-series)
  - [Spatial Pyramid Pooling Related](#spatial-pyramid-pooling-related)
  - [Further Improvements of R-CNN](#further-improvements-of-r-cnn)
- [Single-Stage Detectors](#single-stage-detectors)
  - [unclassified](#unclassified-2)
  - [SSD and its Variants](#ssd-and-its-variants)
  - [YOLO Series](#yolo-series)
  - [Methods using multiple layers (2016, FPN)](#methods-using-multiple-layers-2016-fpn)
  - [Improvements](#improvements)
- [Design of Loss Functions](#design-of-loss-functions)
  - [Focal Loss](#focal-loss)
  - [IoU Loss](#iou-loss)
- [Anchor-Free Frameworks](#anchor-free-frameworks)
- [Neural Architecture Search (EfficientNetV2, 2021)](#neural-architecture-search-efficientnetv2-2021)
- [Attention Mechanism](#attention-mechanism)
- [Transformer Architectures Applied to Detection](#transformer-architectures-applied-to-detection)
- [Weekly-Supervised Learning](#weekly-supervised-learning)

----------------------------------------------------------------------------------------------------

## Unclassified

* [[Multi-Stage Features](https://arxiv.org/abs/1212.0142)]
    [[pdf](https://arxiv.org/pdf/1212.0142.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1212.0142/)]
    * Title: Pedestrian Detection with Unsupervised Multi-Stage Feature Learning
    * Year: 01 Dec `2012`
    * Authors: Pierre Sermanet, Koray Kavukcuoglu, Soumith Chintala, Yann LeCun
    * Abstract: Pedestrian detection is a problem of considerable practical interest. Adding to the list of successful applications of deep learning methods to vision, we report state-of-the-art and competitive results on all major pedestrian datasets with a convolutional network model. The model uses a few new twists, such as multi-stage features, connections that skip layers to integrate global shape information with local distinctive motif information, and an unsupervised method based on convolutional sparse coding to pre-train the filters at each stage.
* [MultiGrasp](https://arxiv.org/abs/1412.3128)
    [[pdf](https://arxiv.org/pdf/1412.3128.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.3128/)]
    * Title: Real-Time Grasp Detection Using Convolutional Neural Networks
    * Year: 09 Dec `2014`
    * Authors: Joseph Redmon, Anelia Angelova
    * Abstract: We present an accurate, real-time approach to robotic grasp detection based on convolutional neural networks. Our network performs single-stage regression to graspable bounding boxes without using standard sliding window or region proposal techniques. The model outperforms state-of-the-art approaches by 14 percentage points and runs at 13 frames per second on a GPU. Our network can simultaneously perform classification so that in a single step it recognizes the object and finds a good grasp rectangle. A modification to this model predicts multiple grasps per object by using a locally constrained prediction mechanism. The locally constrained model performs significantly better, especially on objects that can be grasped in a variety of ways.
<!-- * [DPM](https://arxiv.org/abs/1409.5403) -->
<!-- * Title: Deformable Part Models are Convolutional Neural Networks -->
<!-- * Year: 18 Sep `2014` -->
<!-- * Author: Ross Girshick -->
<!-- * Abstract: Deformable part models (DPMs) and convolutional neural networks (CNNs) are two widely used tools for visual recognition. They are typically viewed as distinct approaches: DPMs are graphical models (Markov random fields), while CNNs are "black-box" non-linear classifiers. In this paper, we show that a DPM can be formulated as a CNN, thus providing a novel synthesis of the two ideas. Our construction involves unrolling the DPM inference algorithm and mapping each step to an equivalent (and at times novel) CNN layer. From this perspective, it becomes natural to replace the standard image features used in DPM with a learned feature extractor. We call the resulting model DeepPyramid DPM and experimentally validate it on PASCAL VOC. DeepPyramid DPM significantly outperforms DPMs based on histograms of oriented gradients features (HOG) and slightly outperforms a comparable version of the recently introduced R-CNN detection system, while running an order of magnitude faster. -->
* [[Multi-Region CNN](https://arxiv.org/abs/1505.01749)]
    [[pdf](https://arxiv.org/pdf/1505.01749.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1505.01749/)]
    * Title: Object detection via a multi-region & semantic segmentation-aware CNN model
    * Year: 07 May `2015`
    * Authors: Spyros Gidaris, Nikos Komodakis
    * Abstract: We propose an object detection system that relies on a multi-region deep convolutional neural network (CNN) that also encodes semantic segmentation-aware features. The resulting CNN-based representation aims at capturing a diverse set of discriminative appearance factors and exhibits localization sensitivity that is essential for accurate object localization. We exploit the above properties of our recognition module by integrating it on an iterative localization mechanism that alternates between scoring a box proposal and refining its location with a deep CNN regression model. Thanks to the efficient use of our modules, we detect objects with very high localization accuracy. On the detection challenges of PASCAL VOC2007 and PASCAL VOC2012 we achieve mAP of 78.2% and 73.9% correspondingly, surpassing any other published work by a significant margin.
* [[DeepProposal](https://arxiv.org/abs/1510.04445)]
    [[pdf](https://arxiv.org/pdf/1510.04445.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1510.04445/)]
    * Title: DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers
    * Year: 15 Oct `2015`
    * Authors: Amir Ghodrati, Ali Diba, Marco Pedersoli, Tinne Tuytelaars, Luc Van Gool
    * Abstract: In this paper we evaluate the quality of the activation layers of a convolutional neural network (CNN) for the gen- eration of object proposals. We generate hypotheses in a sliding-window fashion over different activation layers and show that the final convolutional layers can find the object of interest with high recall but poor localization due to the coarseness of the feature maps. Instead, the first layers of the network can better localize the object of interest but with a reduced recall. Based on this observation we design a method for proposing object locations that is based on CNN features and that combines the best of both worlds. We build an inverse cascade that, going from the final to the initial convolutional layers of the CNN, selects the most promising object locations and refines their boxes in a coarse-to-fine manner. The method is efficient, because i) it uses the same features extracted for detection, ii) it aggregates features using integral images, and iii) it avoids a dense evaluation of the proposals due to the inverse coarse-to-fine cascade. The method is also accurate; it outperforms most of the previously proposed object proposals approaches and when plugged into a CNN-based detector produces state-of-the- art detection performance.
* [[Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)]
    [[pdf](https://arxiv.org/pdf/1901.01892.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1901.01892/)]
    * Title: Scale-Aware Trident Networks for Object Detection
    * Year: 07 Jan `2019`
    * Authors: Yanghao Li, Yuntao Chen, Naiyan Wang, Zhaoxiang Zhang
    * Abstract: Scale variation is one of the key challenges in object detection. In this work, we first present a controlled experiment to investigate the effect of receptive fields for scale variation in object detection. Based on the findings from the exploration experiments, we propose a novel Trident Network (TridentNet) aiming to generate scale-specific feature maps with a uniform representational power. We construct a parallel multi-branch architecture in which each branch shares the same transformation parameters but with different receptive fields. Then, we adopt a scale-aware training scheme to specialize each branch by sampling object instances of proper scales for training. As a bonus, a fast approximation version of TridentNet could achieve significant improvements without any additional parameters and computational cost compared with the vanilla detector. On the COCO dataset, our TridentNet with ResNet-101 backbone achieves state-of-the-art single-model results of 48.4 mAP. Codes are available at this https URL.
* [[CenterNet](https://arxiv.org/abs/1904.07850)]
    [[pdf](https://arxiv.org/pdf/1904.07850.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.07850/)]
    * Title: Objects as Points
    * Year: 16 Apr `2019`
    * Authors: Xingyi Zhou, Dequan Wang, Philipp Krähenbühl
    * Abstract: Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.
* [[DetectoRS](https://arxiv.org/abs/2006.02334)]
    [[pdf](https://arxiv.org/pdf/2006.02334.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2006.02334/)]
    * Title: DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution
    * Year: 03 Jun `2020`
    * Authors: Siyuan Qiao, Liang-Chieh Chen, Alan Yuille
    * Abstract: Many modern object detectors demonstrate outstanding performances by using the mechanism of looking and thinking twice. In this paper, we explore this mechanism in the backbone design for object detection. At the macro level, we propose Recursive Feature Pyramid, which incorporates extra feedback connections from Feature Pyramid Networks into the bottom-up backbone layers. At the micro level, we propose Switchable Atrous Convolution, which convolves the features with different atrous rates and gathers the results using switch functions. Combining them results in DetectoRS, which significantly improves the performances of object detection. On COCO test-dev, DetectoRS achieves state-of-the-art 55.7% box AP for object detection, 48.5% mask AP for instance segmentation, and 50.0% PQ for panoptic segmentation. The code is made publicly available.
* [[Robust and Accurate Object Detection via Adversarial Learning](https://arxiv.org/abs/2103.13886)]
    [[pdf](https://arxiv.org/pdf/2103.13886.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.13886/)]
    * Title: Robust and Accurate Object Detection via Adversarial Learning
    * Year: 23 Mar `2021`
    * Authors: Xiangning Chen, Cihang Xie, Mingxing Tan, Li Zhang, Cho-Jui Hsieh, Boqing Gong
    * Abstract: Data augmentation has become a de facto component for training high-performance deep image classifiers, but its potential is under-explored for object detection. Noting that most state-of-the-art object detectors benefit from fine-tuning a pre-trained classifier, we first study how the classifiers' gains from various data augmentations transfer to object detection. The results are discouraging; the gains diminish after fine-tuning in terms of either accuracy or robustness. This work instead augments the fine-tuning stage for object detectors by exploring adversarial examples, which can be viewed as a model-dependent data augmentation. Our method dynamically selects the stronger adversarial images sourced from a detector's classification and localization branches and evolves with the detector to ensure the augmentation policy stays current and relevant. This model-dependent augmentation generalizes to different object detectors better than AutoAugment, a model-agnostic augmentation policy searched based on one particular detector. Our approach boosts the performance of state-of-the-art EfficientDets by +1.1 mAP on the COCO object detection benchmark. It also improves the detectors' robustness against natural distortions by +3.8 mAP and against domain shift by +1.3 mAP. Models are available at this https URL
* [[An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189)]
    [[pdf](https://arxiv.org/pdf/1711.08189.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1711.08189/)]
    * Title: An Analysis of Scale Invariance in Object Detection - SNIP
    * Year: 22 Nov `2017`
    * Authors: Bharat Singh, Larry S. Davis
    * Abstract: An analysis of different techniques for recognizing and detecting objects under extreme scale variation is presented. Scale specific and scale invariant design of detectors are compared by training them with different configurations of input data. By evaluating the performance of different network architectures for classifying small objects on ImageNet, we show that CNNs are not robust to changes in scale. Based on this analysis, we propose to train and test detectors on the same scales of an image-pyramid. Since small and large objects are difficult to recognize at smaller and larger scales respectively, we present a novel training scheme called Scale Normalization for Image Pyramids (SNIP) which selectively back-propagates the gradients of object instances of different sizes as a function of the image scale. On the COCO dataset, our single model performance is 45.7% and an ensemble of 3 networks obtains an mAP of 48.3%. We use off-the-shelf ImageNet-1000 pre-trained models and only train with bounding box supervision. Our submission won the Best Student Entry in the COCO 2017 challenge. Code will be made available at \url{this http URL}.
* [[SNIPER](https://arxiv.org/abs/1805.09300)]
    [[pdf](https://arxiv.org/pdf/1805.09300.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1805.09300/)]
    * Title: SNIPER: Efficient Multi-Scale Training
    * Year: 23 May `2018`
    * Authors: Bharat Singh, Mahyar Najibi, Larry S. Davis
    * Abstract: We present SNIPER, an algorithm for performing efficient multi-scale training in instance level visual recognition tasks. Instead of processing every pixel in an image pyramid, SNIPER processes context regions around ground-truth instances (referred to as chips) at the appropriate scale. For background sampling, these context-regions are generated using proposals extracted from a region proposal network trained with a short learning schedule. Hence, the number of chips generated per image during training adaptively changes based on the scene complexity. SNIPER only processes 30% more pixels compared to the commonly used single scale training at 800x1333 pixels on the COCO dataset. But, it also observes samples from extreme resolutions of the image pyramid, like 1400x2000 pixels. As SNIPER operates on resampled low resolution chips (512x512 pixels), it can have a batch size as large as 20 on a single GPU even with a ResNet-101 backbone. Therefore it can benefit from batch-normalization during training without the need for synchronizing batch-normalization statistics across GPUs. SNIPER brings training of instance level recognition tasks like object detection closer to the protocol for image classification and suggests that the commonly accepted guideline that it is important to train on high resolution images for instance level visual recognition tasks might not be correct. Our implementation based on Faster-RCNN with a ResNet-101 backbone obtains an mAP of 47.6% on the COCO dataset for bounding box detection and can process 5 images per second during inference with a single GPU. Code is available at this https URL.
* [[MobileDets](https://arxiv.org/abs/2004.14525)]
    [[pdf](https://arxiv.org/pdf/2004.14525.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2004.14525/)]
    * Title: MobileDets: Searching for Object Detection Architectures for Mobile Accelerators
    * Year: 30 Apr `2020`
    * Authors: Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin, Gabriel Bender, Yongzhe Wang, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh, Bo Chen
    * Abstract: Inverted bottleneck layers, which are built upon depthwise convolutions, have been the predominant building blocks in state-of-the-art object detection models on mobile devices. In this work, we investigate the optimality of this design pattern over a broad range of mobile accelerators by revisiting the usefulness of regular convolutions. We discover that regular convolutions are a potent component to boost the latency-accuracy trade-off for object detection on accelerators, provided that they are placed strategically in the network via neural architecture search. By incorporating regular convolutions in the search space and directly optimizing the network architectures for object detection, we obtain a family of object detection models, MobileDets, that achieve state-of-the-art results across mobile accelerators. On the COCO object detection task, MobileDets outperform MobileNetV3+SSDLite by 1.7 mAP at comparable mobile CPU inference latencies. MobileDets also outperform MobileNetV2+SSDLite by 1.9 mAP on mobile CPUs, 3.7 mAP on Google EdgeTPU, 3.4 mAP on Qualcomm Hexagon DSP and 2.7 mAP on Nvidia Jetson GPU without increasing latency. Moreover, MobileDets are comparable with the state-of-the-art MnasFPN on mobile CPUs even without using the feature pyramid, and achieve better mAP scores on both EdgeTPUs and DSPs with up to 2x speedup. Code and models are available in the TensorFlow Object Detection API: this https URL.
* [[Object Detectors Emerge in Deep Scene CNNs](https://arxiv.org/abs/1412.6856)]
    [[pdf](https://arxiv.org/pdf/1412.6856.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.6856/)]
    * Title: Object Detectors Emerge in Deep Scene CNNs
    * Year: 22 Dec `2014`
    * Authors: Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba
    * Abstract: With the success of new computational architectures for visual processing, such as convolutional neural networks (CNN) and access to image databases with millions of labeled examples (e.g., ImageNet, Places), the state of the art in computer vision is advancing rapidly. One important factor for continued progress is to understand the representations that are learned by the inner layers of these deep architectures. Here we show that object detectors emerge from training CNNs to perform scene classification. As scenes are composed of objects, the CNN for scene classification automatically discovers meaningful objects detectors, representative of the learned scene categories. With object detectors emerging as a result of learning to recognize scenes, our work demonstrates that the same network can perform both scene recognition and object localization in a single forward-pass, without ever having been explicitly taught the notion of objects.
    * Comments:
        * > Although theoretically the receptive field of ResNet [13] is already larger than the input image, it is shown by Zhou et al. [42] that the empirical receptive field of CNN is much smaller than the theoretical one especially on high-level layers. (2016, PSPNet)

## Localization

* [[Spatial Dropout](https://arxiv.org/abs/1411.4280)]
    [[pdf](https://arxiv.org/pdf/1411.4280.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1411.4280/)]
    * Title: Efficient Object Localization Using Convolutional Networks
    * Year: 16 Nov `2014`
    * Authors: Jonathan Tompson, Ross Goroshin, Arjun Jain, Yann LeCun, Christopher Bregler
    * Institutions: [New York University]
    * Abstract: Recent state-of-the-art performance on human-body pose estimation has been achieved with Deep Convolutional Networks (ConvNets). Traditional ConvNet architectures include pooling and sub-sampling layers which reduce computational requirements, introduce invariance and prevent over-training. These benefits of pooling come at the cost of reduced localization accuracy. We introduce a novel architecture which includes an efficient `position refinement' model that is trained to estimate the joint offset location within a small region of the image. This refinement model is jointly trained in cascade with a state-of-the-art ConvNet model to achieve improved accuracy in human joint location estimation. We show that the variance of our detector approaches the variance of human annotations on the FLIC dataset and outperforms all existing approaches on the MPII-human-pose dataset.

## Sliding-Window Approaches

* [[HOG](https://ieeexplore.ieee.org/document/1467360)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1467360)]
    * Title: Histograms of oriented gradients for human detection
    * Year: 25 July `2005`
    * Authors: N. Dalal; B. Triggs
    * Institutions: [INRIA Rhone-Alps]
    * Abstract: We study the question of feature sets for robust visual object recognition; adopting linear SVM based human detection as a test case. After reviewing existing edge and gradient based descriptors, we show experimentally that grids of histograms of oriented gradient (HOG) descriptors significantly outperform existing feature sets for human detection. We study the influence of each stage of the computation on performance, concluding that fine-scale gradients, fine orientation binning, relatively coarse spatial binning, and high-quality local contrast normalization in overlapping descriptor blocks are all important for good results. The new approach gives near-perfect separation on the original MIT pedestrian database, so we introduce a more challenging dataset containing over 1800 annotated human images with a large range of pose variations and backgrounds.
* [DPM](https://ieeexplore.ieee.org/document/5255236)
    * Title: Object Detection with Discriminatively Trained Part-Based Models
    * Year: `2010`
    * Author: Pedro F. Felzenszwalb
    * Abstract: We describe an object detection system based on mixtures of multiscale deformable part models. Our system is able to represent highly variable object classes and achieves state-of-the-art results in the PASCAL object detection challenges. While deformable part models have become quite popular, their value had not been demonstrated on difficult benchmarks such as the PASCAL data sets. Our system relies on new methods for discriminative training with partially labeled data. We combine a margin-sensitive approach for data-mining hard negative examples with a formalism we call latent SVM. A latent SVM is a reformulation of MI--SVM in terms of latent variables. A latent SVM is semiconvex, and the training problem becomes convex once latent information is specified for the positive examples. This leads to an iterative training algorithm that alternates between fixing latent values for positive examples and optimizing the latent SVM objective function.
    * Comments:
        * > (2016, FPN) DPM [7] required dense scale sampling to achieve good results (e.g., 10 scales per octave).
* [[OverFeat](https://arxiv.org/abs/1312.6229)]
    [[pdf](https://arxiv.org/pdf/1312.6229.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1312.6229/)]
    * Title: OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
    * Year: 21 Dec `2013`
    * Authors: Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, Yann LeCun
    * Institutions: [Courant Institute of Mathematical Sciences, New York University]
    * Abstract: We present an integrated framework for using Convolutional Networks for classification, localization and detection. We show how a multiscale and sliding window approach can be efficiently implemented within a ConvNet. We also introduce a novel deep learning approach to localization by learning to predict object boundaries. Bounding boxes are then accumulated rather than suppressed in order to increase detection confidence. We show that different tasks can be learned simultaneously using a single shared network. This integrated framework is the winner of the localization task of the ImageNet Large Scale Visual Recognition Challenge 2013 (ILSVRC2013) and obtained very competitive results for the detection and classifications tasks. In post-competition work, we establish a new state of the art for the detection task. Finally, we release a feature extractor from our best model called OverFeat.
    * Comments:
        * > (2013, R-CNN) OverFeat uses a sliding-window CNN for detection and until now was the best performing method on ILSVRC2013 detection.
        * > (2013, R-CNN) OverFeat can be seen (roughly) as a special case of R-CNN. If one were to replace selective search region proposals with a multi-scale pyramid of regular square regions and change the per-class bounding-box regressors to a single bounding-box regressor, then the systems would be very similar (modulo some potentially significant differences in how they are trained: CNN detection fine-tuning, using SVMs, etc.).
        * > (2016, FPN) OverFeat adopted a strategy similar to early neural network face detectors by applying a ConvNet as a sliding window detector on an image pyramid.

## Object Proposal Approaches (DeepMask, 2015)

Survey

* [[What makes for effective detection proposals?](https://arxiv.org/abs/1502.05082)]
    [[pdf](https://arxiv.org/pdf/1502.05082.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1502.05082/)]
    * Title: What makes for effective detection proposals?
    * Year: 17 Feb `2015`
    * Authors: Jan Hosang, Rodrigo Benenson, Piotr Dollár, Bernt Schiele
    * Abstract: Current top performing object detectors employ detection proposals to guide the search for objects, thereby avoiding exhaustive sliding window search across images. Despite the popularity and widespread use of detection proposals, it is unclear which trade-offs are made when using them during object detection. We provide an in-depth analysis of twelve proposal methods along with four baselines regarding proposal repeatability, ground truth annotation recall on PASCAL, ImageNet, and MS COCO, and their impact on DPM, R-CNN, and Fast R-CNN detection performance. Our analysis shows that for object detection improving proposal localisation accuracy is as important as improving recall. We introduce a novel metric, the average recall (AR), which rewards both high recall and good localisation and correlates surprisingly well with detection performance. Our findings show common strengths and weaknesses of existing methods, and provide insights and metrics for selecting and tuning proposal methods.

> Most object proposal approaches leverage low-level grouping and saliency cues. These approaches usually fall into three categories: (1) objectness scoring [1, 34], in which proposals are extracted by measuring the objectness score of bounding boxes, (2) seed segmentation [14, 16, 17], where models start with multiple seed regions and generate separate foreground-background segmentation for each seed, and (3) superpixel merging [31, 24], where multiple over-segmentations are merged according to various heuristics.

objectness scoring

* [Measuring the Objectness of Image Windows](https://ieeexplore.ieee.org/document/6133291)
    * Title: Measuring the Objectness of Image Windows
* [Edge Boxes](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.453.5208)
    * Title: Edge Boxes: Locating Object Proposals from Edges

seed segmentation

* [RIGOR](https://ieeexplore.ieee.org/document/6909444)
    * Title: RIGOR: Reusing Inference in Graph Cuts for Generating Object Regions
* [Geodesic Object Proposals](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_47)
    * Title: Geodesic Object Proposals
    * Authors: Philipp Krähenbühl & Vladlen Koltun
* [Learning to propose objects](https://ieeexplore.ieee.org/abstract/document/7298765)
    * Title: Learning to propose objects
    * Authors: Philipp Krähenbühl; Vladlen Koltun

superpixel merging

* [Selective Search](https://link.springer.com/article/10.1007/s11263-013-0620-5)
    * Title: Selective Search for Object Recognition
    * Year: 02 April `2013`
    * Authors: J. R. R. Uijlings, K. E. A. van de Sande, T. Gevers & A. W. M. Smeulders
    * Institutions: [University of Trento, Trento, Italy], [University of Amsterdam, Amsterdam, The Netherlands]
    * Abstract: This paper addresses the problem of generating possible object locations for use in object recognition. We introduce selective search which combines the strength of both an exhaustive search and segmentation. Like segmentation, we use the image structure to guide our sampling process. Like exhaustive search, we aim to capture all possible object locations. Instead of a single technique to generate possible object locations, we diversify our search and use a variety of complementary image partitionings to deal with as many image conditions as possible. Our selective search results in a small set of data-driven, class-independent, high quality locations, yielding 99 % recall and a Mean Average Best Overlap of 0.879 at 10,097 locations. The reduced number of locations compared to an exhaustive search enables the use of stronger machine learning techniques and stronger appearance models for object recognition. In this paper we show that our selective search enables the use of the powerful Bag-of-Words model for recognition. The selective search software is made publicly available (Software: http://disi.unitn.it/~uijlings/SelectiveSearch.html).
* [[Multiscale Combinatorial Grouping (MCG)](https://arxiv.org/abs/1503.00848)]
    [[pdf](https://arxiv.org/pdf/1503.00848.pdf)]
    [vanity]
    * Title: Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation
    * Year: 03 Mar `2015`
    * Authors: Jordi Pont-Tuset, Pablo Arbelaez, Jonathan T. Barron, Ferran Marques, Jitendra Malik
    * Institutions: [Department of Signal Theory and Communications, Universitat Politecnica de Catalunya, BarcelonaTech (UPC), Spain], [Department of Biomedical Engineering, Universidad de los Andes, Colombia], [ Department of Electrical Engineering and Computer Science, University of California at Berkeley, Berkeley]
    * Abstract: We propose a unified approach for bottom-up hierarchical image segmentation and object proposal generation for recognition, called Multiscale Combinatorial Grouping (MCG). For this purpose, we first develop a fast normalized cuts algorithm. We then propose a high-performance hierarchical segmenter that makes effective use of multiscale information. Finally, we propose a grouping strategy that combines our multiscale regions into highly-accurate object proposals by exploring efficiently their combinatorial space. We also present Single-scale Combinatorial Grouping (SCG), a faster version of MCG that produces competitive proposals in under five second per image. We conduct an extensive and comprehensive empirical validation on the BSDS500, SegVOC12, SBD, and COCO datasets, showing that MCG produces state-of-the-art contours, hierarchical regions, and object proposals.

## Region Proposals (2013, R-CNN)

* Measuring the Objectness of Image Windows
* Selective Search for Object Recognition
* [[Category independent object proposals](https://link.springer.com/chapter/10.1007/978-3-642-15555-0_42)]
    [[pdf](https://link.springer.com/content/pdf/10.1007/978-3-642-15555-0_42.pdf)]
    * Title: Category independent object proposals
    * Authors: Ian Endres & Derek Hoiem
    * Year: `2010`
    * Abstract: We propose a category-independent method to produce a bag of regions and rank them, such that top-ranked regions are likely to be good segmentations of different objects. Our key objectives are completeness and diversity: every object should have at least one good proposed region, and a diverse set should be top-ranked. Our approach is to generate a set of segmentations by performing graph cuts based on a seed region and a learned affinity function. Then, the regions are ranked using structured learning based on various cues. Our experiments on BSDS and PASCAL VOC 2008 demonstrate our ability to find most objects within a small bag of proposed regions.
* [[CPMC](https://ieeexplore.ieee.org/document/6095566)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6095566)]
    * Title: CPMC: Automatic Object Segmentation Using Constrained Parametric Min-Cuts
    * Year: 06 December `2011`
    * Authors: Joao Carreira; Cristian Sminchisescu
    * Abstract: We present a novel framework to generate and rank plausible hypotheses for the spatial extent of objects in images using bottom-up computational processes and mid-level selection cues. The object hypotheses are represented as figure-ground segmentations, and are extracted automatically, without prior knowledge of the properties of individual object classes, by solving a sequence of Constrained Parametric Min-Cut problems (CPMC) on a regular image grid. In a subsequent step, we learn to rank the corresponding segments by training a continuous model to predict how likely they are to exhibit real-world regularities (expressed as putative overlap with ground truth) based on their mid-level region properties, then diversify the estimated overlap score using maximum marginal relevance measures. We show that this algorithm significantly outperforms the state of the art for low-level segmentation in the VOC 2009 and 2010 data sets. In our companion papers [1], [2], we show that the algorithm can be used, successfully, in a segmentation-based visual object category recognition pipeline. This architecture ranked first in the VOC2009 and VOC2010 image segmentation and labeling challenges.

## Two-Stage Detectors

### unclassified

* [[MultiBox](https://arxiv.org/abs/1312.2249)]
    [[pdf](https://arxiv.org/pdf/1312.2249.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1312.2249/)]
    * Title: Scalable Object Detection using Deep Neural Networks
    * Year: 08 Dec `2013`
    * Authors: Dumitru Erhan, Christian Szegedy, Alexander Toshev, Dragomir Anguelov
    * Abstract: Deep convolutional neural networks have recently achieved state-of-the-art performance on a number of image recognition benchmarks, including the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC-2012). The winning model on the localization sub-task was a network that predicts a single bounding box and a confidence score for each object category in the image. Such a model captures the whole-image context around the objects but cannot handle multiple instances of the same object in the image without naively replicating the number of outputs for each instance. In this work, we propose a saliency-inspired neural network model for detection, which predicts a set of class-agnostic bounding boxes along with a single score for each box, corresponding to its likelihood of containing any object of interest. The model naturally handles a variable number of instances for each class and allows for cross-class generalization at the highest levels of the network. We are able to obtain competitive recognition performance on VOC2007 and ILSVRC2012, while using only the top few predicted locations in each image and a small number of neural network evaluations.
    * Comments:
        * > Unlike R-CNN, Szegedy et al. train a convolutional neural network to predict regions of interest [8] instead of using Selective Search. MultiBox can also perform single object detection by replacing the confidence prediction with a single class prediction. However, MultiBox cannot perform general object detection and is still just a piece in a larger detection pipeline, requiring further image patch classification. (YOLOv1, 2015)
* [[MSC-MultiBox](https://arxiv.org/abs/1412.1441)]
    [[pdf](https://arxiv.org/pdf/1412.1441.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.1441/)]
    * Title: Scalable, High-Quality Object Detection
    * Year: 03 Dec `2014`
    * Authors: Christian Szegedy, Scott Reed, Dumitru Erhan, Dragomir Anguelov, Sergey Ioffe
    * Institutions: [Google Inc.], [University of Michigan]
    * Abstract: Current high-quality object detection approaches use the scheme of salience-based object proposal methods followed by post-classification using deep convolutional features. This spurred recent research in improving object proposal methods. However, domain agnostic proposal generation has the principal drawback that the proposals come unranked or with very weak ranking, making it hard to trade-off quality for running time. This raises the more fundamental question of whether high-quality proposal generation requires careful engineering or can be derived just from data alone. We demonstrate that learning-based proposal methods can effectively match the performance of hand-engineered methods while allowing for very efficient runtime-quality trade-offs. Using the multi-scale convolutional MultiBox (MSC-MultiBox) approach, we substantially advance the state-of-the-art on the ILSVRC 2014 detection challenge data set, with $0.5$ mAP for a single model and $0.52$ mAP for an ensemble of two models. MSC-Multibox significantly improves the proposal quality over its predecessor MultiBox~method: AP increases from $0.42$ to $0.53$ for the ILSVRC detection challenge. Finally, we demonstrate improved bounding-box recall compared to Multiscale Combinatorial Grouping with less proposals on the Microsoft-COCO data set.
* [[DeepBox](https://arxiv.org/abs/1505.02146)]
    [[pdf](https://arxiv.org/pdf/1505.02146.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1505.02146/)]
    * Title: DeepBox: Learning Objectness with Convolutional Networks
    * Year: 08 May `2015`
    * Authors: Weicheng Kuo, Bharath Hariharan, Jitendra Malik
    * Abstract: Existing object proposal approaches use primarily bottom-up cues to rank proposals, while we believe that objectness is in fact a high level construct. We argue for a data-driven, semantic approach for ranking object proposals. Our framework, which we call DeepBox, uses convolutional neural networks (CNNs) to rerank proposals from a bottom-up method. We use a novel four-layer CNN architecture that is as good as much larger networks on the task of evaluating objectness while being much faster. We show that DeepBox significantly improves over the bottom-up ranking, achieving the same recall with 500 proposals as achieved by bottom-up methods with 2000. This improvement generalizes to categories the CNN has never seen before and leads to a 4.5-point gain in detection mAP. Our implementation achieves this performance while running at 260 ms per image.
    * Comments:
        * > Deepbox [19] proposed a ConvNet model that learns to rerank proposals generated by EdgeBox, a bottom-up method for bounding box proposals. (DeepMask, 2015)
* [[segDeepM](https://arxiv.org/abs/1502.04275)]
    [[pdf](https://arxiv.org/pdf/1502.04275.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1502.04275/)]
    * Title: segDeepM: Exploiting Segmentation and Context in Deep Neural Networks for Object Detection
    * Year: 15 Feb `2015`
    * Authors: Yukun Zhu, Raquel Urtasun, Ruslan Salakhutdinov, Sanja Fidler
    * Institutions: [University of Toronto]
    * Abstract: In this paper, we propose an approach that exploits object segmentation in order to improve the accuracy of object detection. We frame the problem as inference in a Markov Random Field, in which each detection hypothesis scores object appearance as well as contextual information using Convolutional Neural Networks, and allows the hypothesis to choose and score a segment out of a large pool of accurate object segmentation proposals. This enables the detector to incorporate additional evidence when it is available and thus results in more accurate detections. Our experiments show an improvement of 4.1% in mAP over the R-CNN baseline on PASCAL VOC 2010, and 3.4% over the current state-of-the-art, demonstrating the power of our approach.

### R-CNN Series

* [[R-CNN](https://arxiv.org/abs/1311.2524)]
    [[pdf](https://arxiv.org/pdf/1311.2524.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1311.2524/)]
    * Title: Rich feature hierarchies for accurate object detection and semantic segmentation
    * Year: 11 Nov `2013`
    * Authors: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik
    * Institutions: [UC Berkeley]
    * Abstract: Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012---achieving a mAP of 53.3%. Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also compare R-CNN to OverFeat, a recently proposed sliding-window detector based on a similar CNN architecture. We find that R-CNN outperforms OverFeat by a large margin on the 200-class ILSVRC2013 detection dataset. Source code for the complete system is available at this http URL.
    * Comments:
        * > (2014, Inception-v1/GoogLeNet) R-CNN decomposes the overall detection problem into two subproblems: to first utilize low-level cues such as color and superpixel consistency for potential object proposals in a category-agnostic fashion, and to then use CNN classifiers to identify object categories at those locations. Such a two stage approach leverages the accuracy of bounding box segmentation with low-level cues, as well as the highly powerful classification power of state-of-the-art CNNs.
        * > (2015, DeepMask) Girshick et al. [10] proposed a two-phase approach. First, a rich set of object proposals (i.e., a set of image regions which are likely to contain an object) is generated using a fast (but possibly imprecise) algorithm. Second, a convolutional neural network classifier is applied on each of the proposals. This approach provides a notable gain in object detection accuracy compared to classic sliding window approaches. Since then, most state-of-the-art object detectors for both the PASCAL VOC [7] and ImageNet [5] datasets rely on object proposals as a first preprocessing step [10, 15, 33].
        * > (2015, Fast R-CNN) R-CNN is slow because it performs a ConvNet forward pass for each object proposal, without sharing computation.
        * > (2016, FPN) R-CNN adopted a region proposal-based strategy [37] in which each proposal was scale-normalized before classifying with a ConvNet.
* [[Fast R-CNN](https://arxiv.org/abs/1504.08083)]
    [[pdf](https://arxiv.org/pdf/1504.08083.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1504.08083/)]
    * Title: Fast R-CNN
    * Year: 30 Apr `2015`
    * Authors: Ross Girshick
    * Institutions: [Microsoft Research]
    * Abstract: This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3x faster, tests 10x faster, and is more accurate. Fast R-CNN is implemented in Python and C++ (using Caffe) and is available under the open-source MIT License at this https URL.
    * Comments:
        * > (2016, FPN) Recent and more accurate detection methods like Fast R-CNN [11] and Faster R-CNN [29] advocate using features computed from a single scale, because it offers a good trade-off between accuracy and speed.
* [Faster R-CNN/RPN](https://arxiv.org/abs/1506.01497)
    [[pdf](https://arxiv.org/pdf/1506.01497.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1506.01497/)]
    * Title: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
    * Year: 04 Jun `2015`
    * Authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
    * Abstract: State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.
    * Comments:
        * > (2016, FPN) Recent and more accurate detection methods like Fast R-CNN [11] and Faster R-CNN [29] advocate using features computed from a single scale, because it offers a good trade-off between accuracy and speed.

### Spatial Pyramid Pooling Related

* [[The pyramid match kernel](https://ieeexplore.ieee.org/document/1544890)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1544890)]
    * Title: The pyramid match kernel: discriminative classification with sets of image features
    * Year: 05 December `2005`
    * Authors: K. Grauman; T. Darrell
    * Abstract: Discriminative learning is challenging when examples are sets of features, and the sets vary in cardinality and lack any sort of meaningful ordering. Kernel-based classification methods can learn complex decision boundaries, but a kernel over unordered set inputs must somehow solve for correspondences epsivnerally a computationally expensive task that becomes impractical for large set sizes. We present a new fast kernel function which maps unordered feature sets to multi-resolution histograms and computes a weighted histogram intersection in this space. This "pyramid match" computation is linear in the number of features, and it implicitly finds correspondences based on the finest resolution histogram cell where a matched pair first appears. Since the kernel does not penalize the presence of extra features, it is robust to clutter. We show the kernel function is positive-definite, making it valid for use in learning algorithms whose optimal solutions are guaranteed only for Mercer kernels. We demonstrate our algorithm on object recognition tasks and show it to be accurate and dramatically faster than current approaches.
* [[Spatial Pyramid Matching](https://ieeexplore.ieee.org/document/1641019)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1641019)]
    * Title: Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories
    * Year: 09 October `2006`
    * Authors: S. Lazebnik; C. Schmid; J. Ponce
    * Abstract: This paper presents a method for recognizing scene categories based on approximate global geometric correspondence. This technique works by partitioning the image into increasingly fine sub-regions and computing histograms of local features found inside each sub-region. The resulting "spatial pyramid" is a simple and computationally efficient extension of an orderless bag-of-features image representation, and it shows significantly improved performance on challenging scene categorization tasks. Specifically, our proposed method exceeds the state of the art on the Caltech-101 database and achieves high accuracy on a large database of fifteen natural scene categories. The spatial pyramid framework also offers insights into the success of several recently proposed image descriptions, including Torralba's "gist" and Lowe's SIFT descriptors.
* [[Spatial Pyramid Pooling (SPP)](https://arxiv.org/abs/1406.4729)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1406.4729.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1406.4729/)]
    * Title: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
    * Year: 18 Jun `2014`
    * Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    * Institutions: [Microsoft Research, Beijing, China], [Xi'an Jiaotong University, Xi'an, China], [University of Science and Technology of China, Hefei, China]
    * Abstract: Existing deep convolutional neural networks (CNNs) require a fixed-size (e.g., 224x224) input image. This requirement is "artificial" and may reduce the recognition accuracy for the images or sub-images of an arbitrary size/scale. In this work, we equip the networks with another pooling strategy, "spatial pyramid pooling", to eliminate the above requirement. The new network structure, called SPP-net, can generate a fixed-length representation regardless of image size/scale. Pyramid pooling is also robust to object deformations. With these advantages, SPP-net should in general improve all CNN-based image classification methods. On the ImageNet 2012 dataset, we demonstrate that SPP-net boosts the accuracy of a variety of CNN architectures despite their different designs. On the Pascal VOC 2007 and Caltech101 datasets, SPP-net achieves state-of-the-art classification results using a single full-image representation and no fine-tuning. The power of SPP-net is also significant in object detection. Using SPP-net, we compute the feature maps from the entire image only once, and then pool features in arbitrary regions (sub-images) to generate fixed-length representations for training the detectors. This method avoids repeatedly computing the convolutional features. In processing test images, our method is 24-102x faster than the R-CNN method, while achieving better or comparable accuracy on Pascal VOC 2007. In ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014, our methods rank #2 in object detection and #3 in image classification among all 38 teams. This manuscript also introduces the improvement made for this competition.
    * Comments:
        * > (2014, FCN) He et al. [17] discard the non-convolutional portion of classification nets to make a feature extractor. They combine proposals and spatial pyramid pooling to yield a localized, fixed-length feature for classification. While fast and effective, this hybrid model cannot be learned end-to-end.
        * > (2015, Fast R-CNN) Spatial pyramid pooling networks (SPPnets) [11] were proposed to speed up R-CNN by sharing computation.
        * > (2016, PSPNet) In [12], feature maps in different levels generated by pyramid pooling were finally flattened and concatenated to be fed into a fully connected layer for classification.
        * > (2016, FPN) SPPnet [15] demonstrated that such region-based detectors could be applied much more efficiently on feature maps extracted on a single image scale.
* [[Strip Pooling](https://arxiv.org/abs/2003.13328)]
    [[pdf](https://arxiv.org/pdf/2003.13328.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2003.13328/)]
    * Title: Strip Pooling: Rethinking Spatial Pooling for Scene Parsing
    * Year: 30 Mar `2020`
    * Authors: Qibin Hou, Li Zhang, Ming-Ming Cheng, Jiashi Feng
    * Abstract: Spatial pooling has been proven highly effective in capturing long-range contextual information for pixel-wise prediction tasks, such as scene parsing. In this paper, beyond conventional spatial pooling that usually has a regular shape of NxN, we rethink the formulation of spatial pooling by introducing a new pooling strategy, called strip pooling, which considers a long but narrow kernel, i.e., 1xN or Nx1. Based on strip pooling, we further investigate spatial pooling architecture design by 1) introducing a new strip pooling module that enables backbone networks to efficiently model long-range dependencies, 2) presenting a novel building block with diverse spatial pooling as a core, and 3) systematically comparing the performance of the proposed strip pooling and conventional spatial pooling techniques. Both novel pooling-based designs are lightweight and can serve as an efficient plug-and-play module in existing scene parsing networks. Extensive experiments on popular benchmarks (e.g., ADE20K and Cityscapes) demonstrate that our simple approach establishes new state-of-the-art results. Code is made available at this https URL.

### Further Improvements of R-CNN

* [[R-FCN](https://arxiv.org/abs/1605.06409)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1605.06409.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.06409/)]
    * Title: R-FCN: Object Detection via Region-based Fully Convolutional Networks
    * Year: 20 May `2016`
    * Authors: Jifeng Dai, Yi Li, Kaiming He, Jian Sun
    * Abstract: We present region-based, fully convolutional networks for accurate and efficient object detection. In contrast to previous region-based detectors such as Fast/Faster R-CNN that apply a costly per-region subnetwork hundreds of times, our region-based detector is fully convolutional with almost all computation shared on the entire image. To achieve this goal, we propose position-sensitive score maps to address a dilemma between translation-invariance in image classification and translation-variance in object detection. Our method can thus naturally adopt fully convolutional image classifier backbones, such as the latest Residual Networks (ResNets), for object detection. We show competitive results on the PASCAL VOC datasets (e.g., 83.6% mAP on the 2007 set) with the 101-layer ResNet. Meanwhile, our result is achieved at a test-time speed of 170ms per image, 2.5-20x faster than the Faster R-CNN counterpart. Code is made publicly available at: this https URL
* [[Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection](https://arxiv.org/abs/1604.04693)]
    [[pdf](https://arxiv.org/pdf/1604.04693.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1604.04693/)]
    * Title: Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection
    * Year: 16 Apr `2016`
    * Authors: Yu Xiang, Wongun Choi, Yuanqing Lin, Silvio Savarese
    * Abstract: In CNN-based object detection methods, region proposal becomes a bottleneck when objects exhibit significant scale variation, occlusion or truncation. In addition, these methods mainly focus on 2D object detection and cannot estimate detailed properties of objects. In this paper, we propose subcategory-aware CNNs for object detection. We introduce a novel region proposal network that uses subcategory information to guide the proposal generating process, and a new detection network for joint detection and subcategory classification. By using subcategories related to object pose, we achieve state-of-the-art performance on both detection and pose estimation on commonly used benchmarks.
* [[Mask R-CNN](https://arxiv.org/abs/1703.06870)]
    [[pdf](https://arxiv.org/pdf/1703.06870.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.06870/)]
    * Title: Mask R-CNN
    * Year: 20 Mar `2017`
    * Authors: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
    * Abstract: We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code has been made available at: this https URL
* [[Cascade R-CNN](https://arxiv.org/abs/1712.00726)]
    [[pdf](https://arxiv.org/pdf/1712.00726.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1712.00726/)]
    * Title: Cascade R-CNN: Delving into High Quality Object Detection
    * Year: 03 Dec `2017`
    * Authors: Zhaowei Cai, Nuno Vasconcelos
    * Abstract: In object detection, an intersection over union (IoU) threshold is required to define positives and negatives. An object detector, trained with low IoU threshold, e.g. 0.5, usually produces noisy detections. However, detection performance tends to degrade with increasing the IoU thresholds. Two main factors are responsible for this: 1) overfitting during training, due to exponentially vanishing positive samples, and 2) inference-time mismatch between the IoUs for which the detector is optimal and those of the input hypotheses. A multi-stage object detection architecture, the Cascade R-CNN, is proposed to address these problems. It consists of a sequence of detectors trained with increasing IoU thresholds, to be sequentially more selective against close false positives. The detectors are trained stage by stage, leveraging the observation that the output of a detector is a good distribution for training the next higher quality detector. The resampling of progressively improved hypotheses guarantees that all detectors have a positive set of examples of equivalent size, reducing the overfitting problem. The same cascade procedure is applied at inference, enabling a closer match between the hypotheses and the detector quality of each stage. A simple implementation of the Cascade R-CNN is shown to surpass all single-model object detectors on the challenging COCO dataset. Experiments also show that the Cascade R-CNN is widely applicable across detector architectures, achieving consistent gains independently of the baseline detector strength. The code will be made available at this https URL.
* [[Sparse R-CNN](https://arxiv.org/abs/2011.12450)]
    [[pdf](https://arxiv.org/pdf/2011.12450.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2011.12450/)]
    * Title: Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
    * Year: 25 Nov `2020`
    * Authors: Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, Ping Luo
    * Abstract: We present Sparse R-CNN, a purely sparse method for object detection in images. Existing works on object detection heavily rely on dense object candidates, such as $k$ anchor boxes pre-defined on all grids of image feature map of size $H\times W$. In our method, however, a fixed sparse set of learned object proposals, total length of $N$, are provided to object recognition head to perform classification and location. By eliminating $HWk$ (up to hundreds of thousands) hand-designed object candidates to $N$ (e.g. 100) learnable proposals, Sparse R-CNN completely avoids all efforts related to object candidates design and many-to-one label assignment. More importantly, final predictions are directly output without non-maximum suppression post-procedure. Sparse R-CNN demonstrates accuracy, run-time and training convergence performance on par with the well-established detector baselines on the challenging COCO dataset, e.g., achieving 45.0 AP in standard $3\times$ training schedule and running at 22 fps using ResNet-50 FPN model. We hope our work could inspire re-thinking the convention of dense prior in object detectors. The code is available at: this https URL.
* [Feature Selective Networks](https://arxiv.org/abs/1711.08879)
    [[pdf](https://arxiv.org/pdf/1711.08879.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1711.08879/)]
    * Title: Feature Selective Networks for Object Detection
    * Year: 24 Nov `2017`
    * Authors: Yao Zhai, Jingjing Fu, Yan Lu, Houqiang Li
    * Abstract: Objects for detection usually have distinct characteristics in different sub-regions and different aspect ratios. However, in prevalent two-stage object detection methods, Region-of-Interest (RoI) features are extracted by RoI pooling with little emphasis on these translation-variant feature components. We present feature selective networks to reform the feature representations of RoIs by exploiting their disparities among sub-regions and aspect ratios. Our network produces the sub-region attention bank and aspect ratio attention bank for the whole image. The RoI-based sub-region attention map and aspect ratio attention map are selectively pooled from the banks, and then used to refine the original RoI features for RoI classification. Equipped with a light-weight detection subnetwork, our network gets a consistent boost in detection performance based on general ConvNet backbones (ResNet-101, GoogLeNet and VGG-16). Without bells and whistles, our detectors equipped with ResNet-101 achieve more than 3% mAP improvement compared to counterparts on PASCAL VOC 2007, PASCAL VOC 2012 and MS COCO datasets.
* [[Dynamic R-CNN](https://arxiv.org/abs/2004.06002)]
    [[pdf](https://arxiv.org/pdf/2004.06002.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2004.06002/)]
    * Title: Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training
    * Year: 13 Apr `2020`
    * Authors: Hongkai Zhang, Hong Chang, Bingpeng Ma, Naiyan Wang, Xilin Chen
    * Abstract: Although two-stage object detectors have continuously advanced the state-of-the-art performance in recent years, the training process itself is far from crystal. In this work, we first point out the inconsistency problem between the fixed network settings and the dynamic training procedure, which greatly affects the performance. For example, the fixed label assignment strategy and regression loss function cannot fit the distribution change of proposals and thus are harmful to training high quality detectors. Consequently, we propose Dynamic R-CNN to adjust the label assignment criteria (IoU threshold) and the shape of regression loss function (parameters of SmoothL1 Loss) automatically based on the statistics of proposals during training. This dynamic design makes better use of the training samples and pushes the detector to fit more high quality samples. Specifically, our method improves upon ResNet-50-FPN baseline with 1.9% AP and 5.5% AP$_{90}$ on the MS COCO dataset with no extra overhead. Codes and models are available at this https URL.

## Single-Stage Detectors

### unclassified

* [[PolarMask](https://arxiv.org/abs/1909.13226)]
    [[pdf](https://arxiv.org/pdf/1909.13226.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1909.13226/)]
    * Title: PolarMask: Single Shot Instance Segmentation with Polar Representation
    * Year: 29 Sep `2019`
    * Authors: Enze Xie, Peize Sun, Xiaoge Song, Wenhai Wang, Ding Liang, Chunhua Shen, Ping Luo
    * Abstract: In this paper, we introduce an anchor-box free and single shot instance segmentation method, which is conceptually simple, fully convolutional and can be used as a mask prediction module for instance segmentation, by easily embedding it into most off-the-shelf detection methods. Our method, termed PolarMask, formulates the instance segmentation problem as instance center classification and dense distance regression in a polar coordinate. Moreover, we propose two effective approaches to deal with sampling high-quality center examples and optimization for dense distance regression, respectively, which can significantly improve the performance and simplify the training process. Without any bells and whistles, PolarMask achieves 32.9% in mask mAP with single-model and single-scale training/testing on challenging COCO dataset. For the first time, we demonstrate a much simpler and flexible instance segmentation framework achieving competitive accuracy. We hope that the proposed PolarMask framework can serve as a fundamental and strong baseline for single shot instance segmentation tasks. Code is available at: this http URL.
* [[OneNet](https://arxiv.org/abs/2012.05780)]
    [[pdf](https://arxiv.org/pdf/2012.05780.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2012.05780/)]
    * Title: What Makes for End-to-End Object Detection?
    * Year: 10 Dec `2020`
    * Authors: Peize Sun, Yi Jiang, Enze Xie, Wenqi Shao, Zehuan Yuan, Changhu Wang, Ping Luo
    * Abstract: Object detection has recently achieved a breakthrough for removing the last one non-differentiable component in the pipeline, Non-Maximum Suppression (NMS), and building up an end-to-end system. However, what makes for its one-to-one prediction has not been well understood. In this paper, we first point out that one-to-one positive sample assignment is the key factor, while, one-to-many assignment in previous detectors causes redundant predictions in inference. Second, we surprisingly find that even training with one-to-one assignment, previous detectors still produce redundant predictions. We identify that classification cost in matching cost is the main ingredient: (1) previous detectors only consider location cost, (2) by additionally introducing classification cost, previous detectors immediately produce one-to-one prediction during inference. We introduce the concept of score gap to explore the effect of matching cost. Classification cost enlarges the score gap by choosing positive samples as those of highest score in the training iteration and reducing noisy positive samples brought by only location cost. Finally, we demonstrate the advantages of end-to-end object detection on crowded scenes. The code is available at: \url{this https URL}.
* [[TDM](https://arxiv.org/abs/1612.06851)]
    [[pdf](https://arxiv.org/pdf/1612.06851.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1612.06851/)]
    * Title: Beyond Skip Connections: Top-Down Modulation for Object Detection
    * Year: 20 Dec `2016`
    * Authors: Abhinav Shrivastava, Rahul Sukthankar, Jitendra Malik, Abhinav Gupta
    * Abstract: In recent years, we have seen tremendous progress in the field of object detection. Most of the recent improvements have been achieved by targeting deeper feedforward networks. However, many hard object categories such as bottle, remote, etc. require representation of fine details and not just coarse, semantic representations. But most of these fine details are lost in the early convolutional layers. What we need is a way to incorporate finer details from lower layers into the detection architecture. Skip connections have been proposed to combine high-level and low-level features, but we argue that selecting the right features from low-level requires top-down contextual information. Inspired by the human visual pathway, in this paper we propose top-down modulations as a way to incorporate fine details into the detection framework. Our approach supplements the standard bottom-up, feedforward ConvNet with a top-down modulation (TDM) network, connected using lateral connections. These connections are responsible for the modulation of lower layer filters, and the top-down network handles the selection and integration of contextual information and low-level features. The proposed TDM architecture provides a significant boost on the COCO testdev benchmark, achieving 28.6 AP for VGG16, 35.2 AP for ResNet101, and 37.3 for InceptionResNetv2 network, without any bells and whistles (e.g., multi-scale, iterative box refinement, etc.).
* [[Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012)]
    [[pdf](https://arxiv.org/pdf/1611.10012.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1611.10012/)]
    * Title: Speed/accuracy trade-offs for modern convolutional object detectors
    * Year: 30 Nov `2016`
    * Authors: Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin Murphy
    * Abstract: The goal of this paper is to serve as a guide for selecting a detection architecture that achieves the right speed/memory/accuracy balance for a given application and platform. To this end, we investigate various ways to trade accuracy for speed and memory usage in modern convolutional object detection systems. A number of successful systems have been proposed in recent years, but apples-to-apples comparisons are difficult due to different base feature extractors (e.g., VGG, Residual Networks), different default image resolutions, as well as different hardware and software platforms. We present a unified implementation of the Faster R-CNN [Ren et al., 2015], R-FCN [Dai et al., 2016] and SSD [Liu et al., 2015] systems, which we view as "meta-architectures" and trace out the speed/accuracy trade-off curve created by using alternative feature extractors and varying other critical parameters such as image size within each of these meta-architectures. On one extreme end of this spectrum where speed and memory are critical, we present a detector that achieves real time speeds and can be deployed on a mobile device. On the opposite end in which accuracy is critical, we present a detector that achieves state-of-the-art performance measured on the COCO detection task.
* FCOS: Fully Convolutional One-Stage Object Detection

### SSD and its Variants

* [[SSD](https://arxiv.org/abs/1512.02325)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1512.02325.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1512.02325/)]
    * Title: SSD: Single Shot MultiBox Detector
    * Year: 08 Dec `2015`
    * Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg
    * Abstract: We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For $300\times 300$ input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for $500\times 500$ input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model. Code is available at this https URL .
    * Comments:
        * > (2016, FPN) The Single Shot Detector (SSD) [22] is one of the first attempts at using a ConvNet's pyramidal feature hierarchy as if it were a featurized image pyramid (Fig. 1(c)).
        * > (2016, FPN) SSD [22] and MS-CNN [3] predict objects at multiple layers of the feature hierarchy without combining features or scores.
* [[DSSD](https://arxiv.org/abs/1701.06659)]
    [[pdf](https://arxiv.org/pdf/1701.06659.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1701.06659/)]
    * Title: DSSD : Deconvolutional Single Shot Detector
    * Year: 23 Jan `2017`
    * Authors: Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi, Alexander C. Berg
    * Abstract: The main contribution of this paper is an approach for introducing additional context into state-of-the-art general object detection. To achieve this we first combine a state-of-the-art classifier (Residual-101[14]) with a fast detection framework (SSD[18]). We then augment SSD+Residual-101 with deconvolution layers to introduce additional large-scale context in object detection and improve accuracy, especially for small objects, calling our resulting system DSSD for deconvolutional single shot detector. While these two contributions are easily described at a high-level, a naive implementation does not succeed. Instead we show that carefully adding additional stages of learned transformations, specifically a module for feed-forward connections in deconvolution and a new output module, enables this new approach and forms a potential way forward for further detection research. Results are shown on both PASCAL VOC and COCO detection. Our DSSD with $513 \times 513$ input achieves 81.5% mAP on VOC2007 test, 80.0% mAP on VOC2012 test, and 33.2% mAP on COCO, outperforming a state-of-the-art method R-FCN[3] on each dataset.
* [[FSSD](https://arxiv.org/abs/1712.00960)]
    [[pdf](https://arxiv.org/pdf/1712.00960.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1712.00960/)]
    * Title: FSSD: Feature Fusion Single Shot Multibox Detector
    * Year: 04 Dec `2017`
    * Authors: Zuoxin Li, Fuqiang Zhou
    * Abstract: SSD (Single Shot Multibox Detector) is one of the best object detection algorithms with both high accuracy and fast speed. However, SSD's feature pyramid detection method makes it hard to fuse the features from different scales. In this paper, we proposed FSSD (Feature Fusion Single Shot Multibox Detector), an enhanced SSD with a novel and lightweight feature fusion module which can improve the performance significantly over SSD with just a little speed drop. In the feature fusion module, features from different layers with different scales are concatenated together, followed by some down-sampling blocks to generate new feature pyramid, which will be fed to multibox detectors to predict the final detection results. On the Pascal VOC 2007 test, our network can achieve 82.7 mAP (mean average precision) at the speed of 65.8 FPS (frame per second) with the input size 300$\times$300 using a single Nvidia 1080Ti GPU. In addition, our result on COCO is also better than the conventional SSD with a large margin. Our FSSD outperforms a lot of state-of-the-art object detection algorithms in both aspects of accuracy and speed. Code is available at this https URL.

### YOLO Series

* [[YOLOv1](https://arxiv.org/abs/1506.02640)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1506.02640.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1506.02640/)]
    * Title: You Only Look Once: Unified, Real-Time Object Detection
    * Year: 08 Jun `2015`
    * Authors: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
    * Abstract: We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. Compared to state-of-the-art detection systems, YOLO makes more localization errors but is far less likely to predict false detections where nothing exists. Finally, YOLO learns very general representations of objects. It outperforms all other detection methods, including DPM and R-CNN, by a wide margin when generalizing from natural images to artwork on both the Picasso Dataset and the People-Art Dataset.
* [[YOLOv2](https://arxiv.org/abs/1612.08242)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1612.08242.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1612.08242/)]
    * Title: YOLO9000: Better, Faster, Stronger
    * Year: 25 Dec `2016`
    * Authors: Joseph Redmon, Ali Farhadi
    * Abstract: We introduce YOLO9000, a state-of-the-art, real-time object detection system that can detect over 9000 object categories. First we propose various improvements to the YOLO detection method, both novel and drawn from prior work. The improved model, YOLOv2, is state-of-the-art on standard detection tasks like PASCAL VOC and COCO. At 67 FPS, YOLOv2 gets 76.8 mAP on VOC 2007. At 40 FPS, YOLOv2 gets 78.6 mAP, outperforming state-of-the-art methods like Faster RCNN with ResNet and SSD while still running significantly faster. Finally we propose a method to jointly train on object detection and classification. Using this method we train YOLO9000 simultaneously on the COCO detection dataset and the ImageNet classification dataset. Our joint training allows YOLO9000 to predict detections for object classes that don't have labelled detection data. We validate our approach on the ImageNet detection task. YOLO9000 gets 19.7 mAP on the ImageNet detection validation set despite only having detection data for 44 of the 200 classes. On the 156 classes not in COCO, YOLO9000 gets 16.0 mAP. But YOLO can detect more than just 200 classes; it predicts detections for more than 9000 different object categories. And it still runs in real-time.
* [[YOLOv3](https://arxiv.org/abs/1804.02767)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1804.02767.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1804.02767/)]
    * Title: YOLOv3: An Incremental Improvement
    * Year: 08 Apr `2018`
    * Authors: Joseph Redmon, Ali Farhadi
    * Abstract: We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5 mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster. As always, all the code is online at this https URL
* [[YOLOv4](https://arxiv.org/abs/2004.10934)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2004.10934.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2004.10934/)]
    * Title: YOLOv4: Optimal Speed and Accuracy of Object Detection
    * Year: 23 Apr `2020`
    * Authors: Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
    * Abstract: There are a huge number of features which are said to improve Convolutional Neural Network (CNN) accuracy. Practical testing of combinations of such features on large datasets, and theoretical justification of the result, is required. Some features operate on certain models exclusively and for certain problems exclusively, or only for small-scale datasets; while some features, such as batch-normalization and residual-connections, are applicable to the majority of models, tasks, and datasets. We assume that such universal features include Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT) and Mish-activation. We use new features: WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, CmBN, DropBlock regularization, and CIoU loss, and combine some of them to achieve state-of-the-art results: 43.5% AP (65.7% AP50) for the MS COCO dataset at a realtime speed of ~65 FPS on Tesla V100. Source code is at this https URL
* [[Scaled-YOLOv4](https://arxiv.org/abs/2011.08036)]
    [[pdf](https://arxiv.org/pdf/2011.08036.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2011.08036/)]
    * Title: Scaled-YOLOv4: Scaling Cross Stage Partial Network
    * Year: 16 Nov `2020`
    * Authors: Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
    * Abstract: We show that the YOLOv4 object detection neural network based on the CSP approach, scales both up and down and is applicable to small and large networks while maintaining optimal speed and accuracy. We propose a network scaling approach that modifies not only the depth, width, resolution, but also structure of the network. YOLOv4-large model achieves state-of-the-art results: 55.5% AP (73.4% AP50) for the MS COCO dataset at a speed of ~16 FPS on Tesla V100, while with the test time augmentation, YOLOv4-large achieves 56.0% AP (73.3 AP50). To the best of our knowledge, this is currently the highest accuracy on the COCO dataset among any published work. The YOLOv4-tiny model achieves 22.0% AP (42.0% AP50) at a speed of 443 FPS on RTX 2080Ti, while by using TensorRT, batch size = 4 and FP16-precision the YOLOv4-tiny achieves 1774 FPS.
* [YOLOF](https://arxiv.org/abs/2103.09460)
    [[pdf](https://arxiv.org/pdf/2103.09460.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.09460/)]
    * Title: You Only Look One-level Feature
    * Year: 17 Mar `2021`
    * Authors: Qiang Chen, Yingming Wang, Tong Yang, Xiangyu Zhang, Jian Cheng, Jian Sun
    * Abstract: This paper revisits feature pyramids networks (FPN) for one-stage detectors and points out that the success of FPN is due to its divide-and-conquer solution to the optimization problem in object detection rather than multi-scale feature fusion. From the perspective of optimization, we introduce an alternative way to address the problem instead of adopting the complex feature pyramids - {\em utilizing only one-level feature for detection}. Based on the simple and efficient solution, we present You Only Look One-level Feature (YOLOF). In our method, two key components, Dilated Encoder and Uniform Matching, are proposed and bring considerable improvements. Extensive experiments on the COCO benchmark prove the effectiveness of the proposed model. Our YOLOF achieves comparable results with its feature pyramids counterpart RetinaNet while being $2.5\times$ faster. Without transformer layers, YOLOF can match the performance of DETR in a single-level feature manner with $7\times$ less training epochs. With an image size of $608\times608$, YOLOF achieves 44.3 mAP running at 60 fps on 2080Ti, which is $13\%$ faster than YOLOv4. Code is available at \url{this https URL}.
* [YOLOR](https://arxiv.org/abs/2105.04206)
    [[pdf](https://arxiv.org/pdf/2105.04206.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2105.04206/)]
    * Title: You Only Learn One Representation: Unified Network for Multiple Tasks
    * Year: 10 May `2021`
    * Authors: Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao
    * Abstract: People ``understand'' the world via vision, hearing, tactile, and also the past experience. Human experience can be learned through normal learning (we call it explicit knowledge), or subconsciously (we call it implicit knowledge). These experiences learned through normal learning or subconsciously will be encoded and stored in the brain. Using these abundant experience as a huge database, human beings can effectively process data, even they were unseen beforehand. In this paper, we propose a unified network to encode implicit knowledge and explicit knowledge together, just like the human brain can learn knowledge from normal learning as well as subconsciousness learning. The unified network can generate a unified representation to simultaneously serve various tasks. We can perform kernel space alignment, prediction refinement, and multi-task learning in a convolutional neural network. The results demonstrate that when implicit knowledge is introduced into the neural network, it benefits the performance of all tasks. We further analyze the implicit representation learnt from the proposed unified network, and it shows great capability on catching the physical meaning of different tasks. The source code of this work is at : this https URL.
* [YOLOS](https://arxiv.org/abs/2106.00666)
    [[pdf](https://arxiv.org/pdf/2106.00666.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.00666/)]
    * Title: You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection
    * Year: 01 Jun `2021`
    * Authors: Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu
    * Abstract: Can Transformer perform 2D object- and region-level recognition from a pure sequence-to-sequence perspective with minimal knowledge about the 2D spatial structure? To answer this question, we present You Only Look at One Sequence (YOLOS), a series of object detection models based on the vanilla Vision Transformer with the fewest possible modifications, region priors, as well as inductive biases of the target task. We find that YOLOS pre-trained on the mid-sized ImageNet-1k dataset only can already achieve quite competitive performance on the challenging COCO object detection benchmark, e.g., YOLOS-Base directly adopted from BERT-Base architecture can obtain 42.0 box AP on COCO val. We also discuss the impacts as well as limitations of current pre-train schemes and model scaling strategies for Transformer in vision through YOLOS. Code and pre-trained models are available at this https URL.
* [YOLOX](https://arxiv.org/abs/2107.08430)
    [[pdf](https://arxiv.org/pdf/2107.08430.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2107.08430/)]
    * Title: YOLOX: Exceeding YOLO Series in 2021
    * Year: 18 Jul `2021`
    * Authors: Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun
    * Abstract: In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported. Source code is at this https URL.

### Methods using multiple layers (2016, FPN)

* [[Inside-Outside Net (ION)](https://arxiv.org/abs/1512.04143)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1512.04143.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1512.04143/)]
    * Title: Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
    * Year: 14 Dec `2015`
    * Authors: Sean Bell, C. Lawrence Zitnick, Kavita Bala, Ross Girshick
    * Abstract: It is well known that contextual and multi-scale representations are important for accurate visual recognition. In this paper we present the Inside-Outside Net (ION), an object detector that exploits information both inside and outside the region of interest. Contextual information outside the region of interest is integrated using spatial recurrent neural networks. Inside, we use skip pooling to extract information at multiple scales and levels of abstraction. Through extensive experiments we evaluate the design space and provide readers with an overview of what tricks of the trade are important. ION improves state-of-the-art on PASCAL VOC 2012 object detection from 73.9% to 76.4% mAP. On the new and more challenging MS COCO dataset, we improve state-of-art-the from 19.7% to 33.1% mAP. In the 2015 MS COCO Detection Challenge, our ION model won the Best Student Entry and finished 3rd place overall. As intuition suggests, our detection results provide strong evidence that context and multi-scale representations improve small object detection.
    * Comments:
        * > (2016, FPN) Several other approaches (HyperNet [18], ParseNet [23], and ION [2]) concatenate features of multiple layers before computing predictions, which is equivalent to summing transformed features.
* [[HyperNet](https://arxiv.org/abs/1604.00600)]
    [[pdf](https://arxiv.org/pdf/1604.00600.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1604.00600/)]
    * Title: HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection
    * Year: 03 Apr `2016`
    * Authors: Tao Kong, Anbang Yao, Yurong Chen, Fuchun Sun
    * Abstract: Almost all of the current top-performing object detection networks employ region proposals to guide the search for object instances. State-of-the-art region proposal methods usually need several thousand proposals to get high recall, thus hurting the detection efficiency. Although the latest Region Proposal Network method gets promising detection accuracy with several hundred proposals, it still struggles in small-size object detection and precise localization (e.g., large IoU thresholds), mainly due to the coarseness of its feature maps. In this paper, we present a deep hierarchical network, namely HyperNet, for handling region proposal generation and object detection jointly. Our HyperNet is primarily based on an elaborately designed Hyper Feature which aggregates hierarchical feature maps first and then compresses them into a uniform space. The Hyper Features well incorporate deep but highly semantic, intermediate but really complementary, and shallow but naturally high-resolution features of the image, thus enabling us to construct HyperNet by sharing them both in generating proposals and detecting objects via an end-to-end joint training strategy. For the deep VGG16 model, our method achieves completely leading recall and state-of-the-art object detection accuracy on PASCAL VOC 2007 and 2012 using only 100 proposals per image. It runs with a speed of 5 fps (including all steps) on a GPU, thus having the potential for real-time processing.
    * Comments:
        * > (2016, FPN) Several other approaches (HyperNet [18], ParseNet [23], and ION [2]) concatenate features of multiple layers before computing predictions, which is equivalent to summing transformed features.
* [[MS-CNN](https://arxiv.org/abs/1607.07155)]
    [[pdf](https://arxiv.org/pdf/1607.07155.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1607.07155/)]
    * Title: A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection
    * Year: 25 Jul `2016`
    * Authors: Zhaowei Cai, Quanfu Fan, Rogerio S. Feris, Nuno Vasconcelos
    * Abstract: A unified deep neural network, denoted the multi-scale CNN (MS-CNN), is proposed for fast multi-scale object detection. The MS-CNN consists of a proposal sub-network and a detection sub-network. In the proposal sub-network, detection is performed at multiple output layers, so that receptive fields match objects of different scales. These complementary scale-specific detectors are combined to produce a strong multi-scale object detector. The unified network is learned end-to-end, by optimizing a multi-task loss. Feature upsampling by deconvolution is also explored, as an alternative to input upsampling, to reduce the memory and computation costs. State-of-the-art object detection performance, at up to 15 fps, is reported on datasets, such as KITTI and Caltech, containing a substantial number of small objects.
    * Comments:
        * > (2016, FPN) SSD [22] and MS-CNN [3] predict objects at multiple layers of the feature hierarchy without combining features or scores.
* [[Feature Pyramid Networks (FPN)](https://arxiv.org/abs/1612.03144)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1612.03144.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1612.03144/)]
    * Title: Feature Pyramid Networks for Object Detection
    * Year: 09 Dec `2016`
    * Authors: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
    * Institutions: [Facebook AI Research (FAIR)], [Cornell University and Cornell Tech]
    * Abstract: Feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive. In this paper, we exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A top-down architecture with lateral connections is developed for building high-level semantic feature maps at all scales. This architecture, called a Feature Pyramid Network (FPN), shows significant improvement as a generic feature extractor in several applications. Using FPN in a basic Faster R-CNN system, our method achieves state-of-the-art single-model results on the COCO detection benchmark without bells and whistles, surpassing all existing single-model entries including those from the COCO 2016 challenge winners. In addition, our method can run at 5 FPS on a GPU and thus is a practical and accurate solution to multi-scale object detection. Code will be made publicly available.
    * Comments:
        * > In contrast to 'symmetric' decoders [47], FPN uses a lightweight decoder (see Fig. 5). (Panoptic FPN, 2019)

### Improvements

* [[OHEM](https://arxiv.org/abs/1604.03540)]
    [[pdf](https://arxiv.org/pdf/1604.03540.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1604.03540/)]
    * Title: Training Region-based Object Detectors with Online Hard Example Mining
    * Year: 12 Apr `2016`
    * Authors: Abhinav Shrivastava, Abhinav Gupta, Ross Girshick
    * Abstract: The field of object detection has made significant advances riding on the wave of region-based ConvNets, but their training procedure still includes many heuristics and hyperparameters that are costly to tune. We present a simple yet surprisingly effective online hard example mining (OHEM) algorithm for training region-based ConvNet detectors. Our motivation is the same as it has always been -- detection datasets contain an overwhelming number of easy examples and a small number of hard examples. Automatic selection of these hard examples can make training more effective and efficient. OHEM is a simple and intuitive algorithm that eliminates several heuristics and hyperparameters in common use. But more importantly, it yields consistent and significant boosts in detection performance on benchmarks like PASCAL VOC 2007 and 2012. Its effectiveness increases as datasets become larger and more difficult, as demonstrated by the results on the MS COCO dataset. Moreover, combined with complementary advances in the field, OHEM leads to state-of-the-art results of 78.9% and 76.3% mAP on PASCAL VOC 2007 and 2012 respectively.
* [[PVANET](https://arxiv.org/abs/1608.08021)]
    [[pdf](https://arxiv.org/pdf/1608.08021.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1608.08021/)]
    * Title: PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection
    * Year: 29 Aug `2016`
    * Authors: Kye-Hyeon Kim, Sanghoon Hong, Byungseok Roh, Yeongjae Cheon, Minje Park
    * Abstract: This paper presents how we can achieve the state-of-the-art accuracy in multi-category object detection task while minimizing the computational cost by adapting and combining recent technical innovations. Following the common pipeline of "CNN feature extraction + region proposal + RoI classification", we mainly redesign the feature extraction part, since region proposal part is not computationally expensive and classification part can be efficiently compressed with common techniques like truncated SVD. Our design principle is "less channels with more layers" and adoption of some building blocks including concatenated ReLU, Inception, and HyperNet. The designed network is deep and thin and trained with the help of batch normalization, residual connections, and learning rate scheduling based on plateau detection. We obtained solid results on well-known object detection benchmarks: 83.8% mAP (mean average precision) on VOC2007 and 82.5% mAP on VOC2012 (2nd place), while taking only 750ms/image on Intel i7-6700K CPU with a single core and 46ms/image on NVIDIA Titan X GPU. Theoretically, our network requires only 12.3% of the computational cost compared to ResNet-101, the winner on VOC2012.
* [[RON](https://arxiv.org/abs/1707.01691)]
    [[pdf](https://arxiv.org/pdf/1707.01691.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1707.01691/)]
    * Title: RON: Reverse Connection with Objectness Prior Networks for Object Detection
    * Year: 06 Jul `2017`
    * Authors: Tao Kong, Fuchun Sun, Anbang Yao, Huaping Liu, Ming Lu, Yurong Chen
    * Abstract: We present RON, an efficient and effective framework for generic object detection. Our motivation is to smartly associate the best of the region-based (e.g., Faster R-CNN) and region-free (e.g., SSD) methodologies. Under fully convolutional architecture, RON mainly focuses on two fundamental problems: (a) multi-scale object localization and (b) negative sample mining. To address (a), we design the reverse connection, which enables the network to detect objects on multi-levels of CNNs. To deal with (b), we propose the objectness prior to significantly reduce the searching space of objects. We optimize the reverse connection, objectness prior and object detector jointly by a multi-task loss function, thus RON can directly predict final detection results from all locations of various feature maps. Extensive experiments on the challenging PASCAL VOC 2007, PASCAL VOC 2012 and MS COCO benchmarks demonstrate the competitive performance of RON. Specifically, with VGG-16 and low resolution 384X384 input size, the network gets 81.3% mAP on PASCAL VOC 2007, 80.7% mAP on PASCAL VOC 2012 datasets. Its superiority increases when datasets become larger and more difficult, as demonstrated by the results on the MS COCO dataset. With 1.5G GPU memory at test phase, the speed of the network is 15 FPS, 3X faster than the Faster R-CNN counterpart.
* [[CoupleNet](https://arxiv.org/abs/1708.02863)]
    [[pdf](https://arxiv.org/pdf/1708.02863.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1708.02863/)]
    * Title: CoupleNet: Coupling Global Structure with Local Parts for Object Detection
    * Year: 09 Aug `2017`
    * Authors: Yousong Zhu, Chaoyang Zhao, Jinqiao Wang, Xu Zhao, Yi Wu, Hanqing Lu
    * Abstract: The region-based Convolutional Neural Network (CNN) detectors such as Faster R-CNN or R-FCN have already shown promising results for object detection by combining the region proposal subnetwork and the classification subnetwork together. Although R-FCN has achieved higher detection speed while keeping the detection performance, the global structure information is ignored by the position-sensitive score maps. To fully explore the local and global properties, in this paper, we propose a novel fully convolutional network, named as CoupleNet, to couple the global structure with local parts for object detection. Specifically, the object proposals obtained by the Region Proposal Network (RPN) are fed into the the coupling module which consists of two branches. One branch adopts the position-sensitive RoI (PSRoI) pooling to capture the local part information of the object, while the other employs the RoI pooling to encode the global and context information. Next, we design different coupling strategies and normalization ways to make full use of the complementary advantages between the global and local branches. Extensive experiments demonstrate the effectiveness of our approach. We achieve state-of-the-art results on all three challenging datasets, i.e. a mAP of 82.7% on VOC07, 80.4% on VOC12, and 34.4% on COCO. Codes will be made publicly available.
* [[DSOD](https://arxiv.org/abs/1708.01241)]
    [[pdf](https://arxiv.org/pdf/1708.01241.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1708.01241/)]
    * Title: DSOD: Learning Deeply Supervised Object Detectors from Scratch
    * Year: 03 Aug `2017`
    * Authors: Zhiqiang Shen, Zhuang Liu, Jianguo Li, Yu-Gang Jiang, Yurong Chen, Xiangyang Xue
    * Abstract: We present Deeply Supervised Object Detector (DSOD), a framework that can learn object detectors from scratch. State-of-the-art object objectors rely heavily on the off-the-shelf networks pre-trained on large-scale classification datasets like ImageNet, which incurs learning bias due to the difference on both the loss functions and the category distributions between classification and detection tasks. Model fine-tuning for the detection task could alleviate this bias to some extent but not fundamentally. Besides, transferring pre-trained models from classification to detection between discrepant domains is even more difficult (e.g. RGB to depth images). A better solution to tackle these two critical problems is to train object detectors from scratch, which motivates our proposed DSOD. Previous efforts in this direction mostly failed due to much more complicated loss functions and limited training data in object detection. In DSOD, we contribute a set of design principles for training object detectors from scratch. One of the key findings is that deep supervision, enabled by dense layer-wise connections, plays a critical role in learning a good detector. Combining with several other principles, we develop DSOD following the single-shot detection (SSD) framework. Experiments on PASCAL VOC 2007, 2012 and MS COCO datasets demonstrate that DSOD can achieve better results than the state-of-the-art solutions with much more compact models. For instance, DSOD outperforms SSD on all three benchmarks with real-time detection speed, while requires only 1/2 parameters to SSD and 1/10 parameters to Faster RCNN. Our code and models are available at: this https URL .

## Design of Loss Functions

### Focal Loss

* [[Focal Loss/RetinaNet](https://arxiv.org/abs/1708.02002)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1708.02002.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1708.02002/)]
    * Title: Focal Loss for Dense Object Detection
    * Year: 07 Aug `2017`
    * Authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
    * Abstract: The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors. Code is at: this https URL.
* [[Generalized Focal Loss V1](https://arxiv.org/abs/2006.04388)]
    [[pdf](https://arxiv.org/pdf/2006.04388.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2006.04388/)]
    * Title: Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection
    * Year: 08 Jun `2020`
    * Authors: Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang
    * Abstract: One-stage detector basically formulates object detection as dense classification and localization. The classification is usually optimized by Focal Loss and the box location is commonly learned under Dirac delta distribution. A recent trend for one-stage detectors is to introduce an individual prediction branch to estimate the quality of localization, where the predicted quality facilitates the classification to improve detection performance. This paper delves into the representations of the above three fundamental elements: quality estimation, classification and localization. Two problems are discovered in existing practices, including (1) the inconsistent usage of the quality estimation and classification between training and inference and (2) the inflexible Dirac delta distribution for localization when there is ambiguity and uncertainty in complex scenes. To address the problems, we design new representations for these elements. Specifically, we merge the quality estimation into the class prediction vector to form a joint representation of localization quality and classification, and use a vector to represent arbitrary distribution of box locations. The improved representations eliminate the inconsistency risk and accurately depict the flexible distribution in real data, but contain continuous labels, which is beyond the scope of Focal Loss. We then propose Generalized Focal Loss (GFL) that generalizes Focal Loss from its discrete form to the continuous version for successful optimization. On COCO test-dev, GFL achieves 45.0\% AP using ResNet-101 backbone, surpassing state-of-the-art SAPD (43.5\%) and ATSS (43.6\%) with higher or comparable inference speed, under the same backbone and training settings. Notably, our best model can achieve a single-model single-scale AP of 48.2\%, at 10 FPS on a single 2080Ti GPU. Code and models are available at this https URL.
* [[Generalized Focal Loss V2](https://arxiv.org/abs/2011.12885)]
    [[pdf](https://arxiv.org/pdf/2011.12885.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2011.12885/)]
    * Title: Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection
    * Year: 25 Nov `2020`
    * Authors: Xiang Li, Wenhai Wang, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang
    * Abstract: Localization Quality Estimation (LQE) is crucial and popular in the recent advancement of dense object detectors since it can provide accurate ranking scores that benefit the Non-Maximum Suppression processing and improve detection performance. As a common practice, most existing methods predict LQE scores through vanilla convolutional features shared with object classification or bounding box regression. In this paper, we explore a completely novel and different perspective to perform LQE -- based on the learned distributions of the four parameters of the bounding box. The bounding box distributions are inspired and introduced as "General Distribution" in GFLV1, which describes the uncertainty of the predicted bounding boxes well. Such a property makes the distribution statistics of a bounding box highly correlated to its real localization quality. Specifically, a bounding box distribution with a sharp peak usually corresponds to high localization quality, and vice versa. By leveraging the close correlation between distribution statistics and the real localization quality, we develop a considerably lightweight Distribution-Guided Quality Predictor (DGQP) for reliable LQE based on GFLV1, thus producing GFLV2. To our best knowledge, it is the first attempt in object detection to use a highly relevant, statistical representation to facilitate LQE. Extensive experiments demonstrate the effectiveness of our method. Notably, GFLV2 (ResNet-101) achieves 46.2 AP at 14.6 FPS, surpassing the previous state-of-the-art ATSS baseline (43.6 AP at 14.6 FPS) by absolute 2.6 AP on COCO {\tt test-dev}, without sacrificing the efficiency both in training and inference. Code will be available at this https URL.

### IoU Loss

* [[UnitBox](https://arxiv.org/abs/1608.01471)]
    [[pdf](https://arxiv.org/pdf/1608.01471.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1608.01471/)]
    * Title: UnitBox: An Advanced Object Detection Network
    * Year: 04 Aug `2016`
    * Authors: Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, Thomas Huang
    * Abstract: In present object detection systems, the deep convolutional neural networks (CNNs) are utilized to predict bounding boxes of object candidates, and have gained performance advantages over the traditional region proposal methods. However, existing deep CNN methods assume the object bounds to be four independent variables, which could be regressed by the $\ell_2$ loss separately. Such an oversimplified assumption is contrary to the well-received observation, that those variables are correlated, resulting to less accurate localization. To address the issue, we firstly introduce a novel Intersection over Union ($IoU$) loss function for bounding box prediction, which regresses the four bounds of a predicted box as a whole unit. By taking the advantages of $IoU$ loss and deep fully convolutional networks, the UnitBox is introduced, which performs accurate and efficient localization, shows robust to objects of varied shapes and scales, and converges fast. We apply UnitBox on face detection task and achieve the best performance among all published methods on the FDDB benchmark.
* [[Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)]
    [[pdf](https://arxiv.org/pdf/1902.09630.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1902.09630/)]
    * Title: Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
    * Year: 25 Feb `2019`
    * Authors: Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, Silvio Savarese
    * Abstract: Intersection over Union (IoU) is the most popular evaluation metric used in the object detection benchmarks. However, there is a gap between optimizing the commonly used distance losses for regressing the parameters of a bounding box and maximizing this metric value. The optimal objective for a metric is the metric itself. In the case of axis-aligned 2D bounding boxes, it can be shown that $IoU$ can be directly used as a regression loss. However, $IoU$ has a plateau making it infeasible to optimize in the case of non-overlapping bounding boxes. In this paper, we address the weaknesses of $IoU$ by introducing a generalized version as both a new loss and a new metric. By incorporating this generalized $IoU$ ($GIoU$) as a loss into the state-of-the art object detection frameworks, we show a consistent improvement on their performance using both the standard, $IoU$ based, and new, $GIoU$ based, performance measures on popular object detection benchmarks such as PASCAL VOC and MS COCO.

## Anchor-Free Frameworks

* [[DeNet](https://arxiv.org/abs/1703.10295)]
    [[pdf](https://arxiv.org/pdf/1703.10295.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.10295/)]
    * Title: DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling
    * Year: 30 Mar `2017`
    * Authors: Lachlan Tychsen-Smith, Lars Petersson
    * Abstract: We define the object detection from imagery problem as estimating a very large but extremely sparse bounding box dependent probability distribution. Subsequently we identify a sparse distribution estimation scheme, Directed Sparse Sampling, and employ it in a single end-to-end CNN based detection model. This methodology extends and formalizes previous state-of-the-art detection models with an additional emphasis on high evaluation rates and reduced manual engineering. We introduce two novelties, a corner based region-of-interest estimator and a deconvolution based CNN model. The resulting model is scene adaptive, does not require manually defined reference bounding boxes and produces highly competitive results on MSCOCO, Pascal VOC 2007 and Pascal VOC 2012 with real-time evaluation rates. Further analysis suggests our model performs particularly well when finegrained object localization is desirable. We argue that this advantage stems from the significantly larger set of available regions-of-interest relative to other methods. Source-code is available from: this https URL
* [[PLN](https://arxiv.org/abs/1706.03646)]
    [[pdf](https://arxiv.org/pdf/1706.03646.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1706.03646/)]
    * Title: Point Linking Network for Object Detection
    * Year: 12 Jun `2017`
    * Authors: Xinggang Wang, Kaibing Chen, Zilong Huang, Cong Yao, Wenyu Liu
    * Abstract: Object detection is a core problem in computer vision. With the development of deep ConvNets, the performance of object detectors has been dramatically improved. The deep ConvNets based object detectors mainly focus on regressing the coordinates of bounding box, e.g., Faster-R-CNN, YOLO and SSD. Different from these methods that considering bounding box as a whole, we propose a novel object bounding box representation using points and links and implemented using deep ConvNets, termed as Point Linking Network (PLN). Specifically, we regress the corner/center points of bounding-box and their links using a fully convolutional network; then we map the corner points and their links back to multiple bounding boxes; finally an object detection result is obtained by fusing the multiple bounding boxes. PLN is naturally robust to object occlusion and flexible to object scale variation and aspect ratio variation. In the experiments, PLN with the Inception-v2 model achieves state-of-the-art single-model and single-scale results on the PASCAL VOC 2007, the PASCAL VOC 2012 and the COCO detection benchmarks without bells and whistles. The source code will be released.
* [[CornerNet](https://arxiv.org/abs/1808.01244)]
    [[pdf](https://arxiv.org/pdf/1808.01244.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1808.01244/)]
    * Title: CornerNet: Detecting Objects as Paired Keypoints
    * Year: 03 Aug `2018`
    * Authors: Hei Law, Jia Deng
    * Abstract: We propose CornerNet, a new approach to object detection where we detect an object bounding box as a pair of keypoints, the top-left corner and the bottom-right corner, using a single convolution neural network. By detecting objects as paired keypoints, we eliminate the need for designing a set of anchor boxes commonly used in prior single-stage detectors. In addition to our novel formulation, we introduce corner pooling, a new type of pooling layer that helps the network better localize corners. Experiments show that CornerNet achieves a 42.2% AP on MS COCO, outperforming all existing one-stage detectors.
* [[ExtremeNet](https://arxiv.org/abs/1901.08043)]
    [[pdf](https://arxiv.org/pdf/1901.08043.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1901.08043/)]
    * Title: Bottom-up Object Detection by Grouping Extreme and Center Points
    * Year: 23 Jan `2019`
    * Authors: Xingyi Zhou, Jiacheng Zhuo, Philipp Krähenbühl
    * Abstract: With the advent of deep learning, object detection drifted from a bottom-up to a top-down recognition problem. State of the art algorithms enumerate a near-exhaustive list of object locations and classify each into: object or not. In this paper, we show that bottom-up approaches still perform competitively. We detect four extreme points (top-most, left-most, bottom-most, right-most) and one center point of objects using a standard keypoint estimation network. We group the five keypoints into a bounding box if they are geometrically aligned. Object detection is then a purely appearance-based keypoint estimation problem, without region classification or implicit feature learning. The proposed method performs on-par with the state-of-the-art region based detection methods, with a bounding box AP of 43.2% on COCO test-dev. In addition, our estimated extreme points directly span a coarse octagonal mask, with a COCO Mask AP of 18.9%, much better than the Mask AP of vanilla bounding boxes. Extreme point guided segmentation further improves this to 34.6% Mask AP.
* [FCOS](https://arxiv.org/abs/1904.01355)
    [[pdf](https://arxiv.org/pdf/1904.01355.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.01355/)]
    * Title: FCOS: Fully Convolutional One-Stage Object Detection
    * Year: 02 Apr `2019`
    * Authors: Zhi Tian, Chunhua Shen, Hao Chen, Tong He
    * Abstract: We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7% in AP with single-model and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler. For the first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks. Code is available at:Code is available at: this https URL
* [[CenterNet](https://arxiv.org/abs/1904.08189)]
    [[pdf](https://arxiv.org/pdf/1904.08189.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.08189/)]
    * Title: CenterNet: Keypoint Triplets for Object Detection
    * Year: 17 Apr `2019`
    * Authors: Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian
    * Abstract: In object detection, keypoint-based approaches often suffer a large number of incorrect object bounding boxes, arguably due to the lack of an additional look into the cropped regions. This paper presents an efficient solution which explores the visual patterns within each cropped region with minimal costs. We build our framework upon a representative one-stage keypoint-based detector named CornerNet. Our approach, named CenterNet, detects each object as a triplet, rather than a pair, of keypoints, which improves both precision and recall. Accordingly, we design two customized modules named cascade corner pooling and center pooling, which play the roles of enriching information collected by both top-left and bottom-right corners and providing more recognizable information at the central regions, respectively. On the MS-COCO dataset, CenterNet achieves an AP of 47.0%, which outperforms all existing one-stage detectors by at least 4.9%. Meanwhile, with a faster inference speed, CenterNet demonstrates quite comparable performance to the top-ranked two-stage detectors. Code is available at this https URL.
* [[FSAF](https://arxiv.org/abs/1903.00621)]
    [[pdf](https://arxiv.org/pdf/1903.00621.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1903.00621/)]
    * Title: Feature Selective Anchor-Free Module for Single-Shot Object Detection
    * Year: 02 Mar `2019`
    * Authors: Chenchen Zhu, Yihui He, Marios Savvides
    * Abstract: We motivate and present feature selective anchor-free (FSAF) module, a simple and effective building block for single-shot object detectors. It can be plugged into single-shot detectors with feature pyramid structure. The FSAF module addresses two limitations brought up by the conventional anchor-based detection: 1) heuristic-guided feature selection; 2) overlap-based anchor sampling. The general concept of the FSAF module is online feature selection applied to the training of multi-level anchor-free branches. Specifically, an anchor-free branch is attached to each level of the feature pyramid, allowing box encoding and decoding in the anchor-free manner at an arbitrary level. During training, we dynamically assign each instance to the most suitable feature level. At the time of inference, the FSAF module can work jointly with anchor-based branches by outputting predictions in parallel. We instantiate this concept with simple implementations of anchor-free branches and online feature selection strategy. Experimental results on the COCO detection track show that our FSAF module performs better than anchor-based counterparts while being faster. When working jointly with anchor-based branches, the FSAF module robustly improves the baseline RetinaNet by a large margin under various settings, while introducing nearly free inference overhead. And the resulting best model can achieve a state-of-the-art 44.6% mAP, outperforming all existing single-shot detectors on COCO.
* [[NAS-FPN](https://arxiv.org/abs/1904.07392)]
    [[pdf](https://arxiv.org/pdf/1904.07392.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.07392/)]
    * Title: NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection
    * Year: 16 Apr `2019`
    * Authors: Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le
    * Abstract: Current state-of-the-art convolutional architectures for object detection are manually designed. Here we aim to learn a better architecture of feature pyramid network for object detection. We adopt Neural Architecture Search and discover a new feature pyramid architecture in a novel scalable search space covering all cross-scale connections. The discovered architecture, named NAS-FPN, consists of a combination of top-down and bottom-up connections to fuse features across scales. NAS-FPN, combined with various backbone models in the RetinaNet framework, achieves better accuracy and latency tradeoff compared to state-of-the-art object detection models. NAS-FPN improves mobile detection accuracy by 2 AP compared to state-of-the-art SSDLite with MobileNetV2 model in [32] and achieves 48.3 AP which surpasses Mask R-CNN [10] detection accuracy with less computation time.
* [[DetNAS](https://arxiv.org/abs/1903.10979)]
    [[pdf](https://arxiv.org/pdf/1903.10979.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1903.10979/)]
    * Title: DetNAS: Backbone Search for Object Detection
    * Year: 26 Mar `2019`
    * Authors: Yukang Chen, Tong Yang, Xiangyu Zhang, Gaofeng Meng, Xinyu Xiao, Jian Sun
    * Abstract: Object detectors are usually equipped with backbone networks designed for image classification. It might be sub-optimal because of the gap between the tasks of image classification and object detection. In this work, we present DetNAS to use Neural Architecture Search (NAS) for the design of better backbones for object detection. It is non-trivial because detection training typically needs ImageNet pre-training while NAS systems require accuracies on the target detection task as supervisory signals. Based on the technique of one-shot supernet, which contains all possible networks in the search space, we propose a framework for backbone search on object detection. We train the supernet under the typical detector training schedule: ImageNet pre-training and detection fine-tuning. Then, the architecture search is performed on the trained supernet, using the detection task as the guidance. This framework makes NAS on backbones very efficient. In experiments, we show the effectiveness of DetNAS on various detectors, for instance, one-stage RetinaNet and the two-stage FPN. We empirically find that networks searched on object detection shows consistent superiority compared to those searched on ImageNet classification. The resulting architecture achieves superior performance than hand-crafted networks on COCO with much less FLOPs complexity.

## Neural Architecture Search (EfficientNetV2, 2021)

* [[EfficientDet](https://arxiv.org/abs/1911.09070)]
    [[pdf](https://arxiv.org/pdf/1911.09070.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1911.09070/)]
    * Title: EfficientDet: Scalable and Efficient Object Detection
    * Year: 20 Nov `2019`
    * Authors: Mingxing Tan, Ruoming Pang, Quoc V. Le
    * Abstract: Model efficiency has become increasingly important in computer vision. In this paper, we systematically study neural network architecture design choices for object detection and propose several key optimizations to improve efficiency. First, we propose a weighted bi-directional feature pyramid network (BiFPN), which allows easy and fast multiscale feature fusion; Second, we propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time. Based on these optimizations and better backbones, we have developed a new family of object detectors, called EfficientDet, which consistently achieve much better efficiency than prior art across a wide spectrum of resource constraints. In particular, with single model and single-scale, our EfficientDet-D7 achieves state-of-the-art 55.1 AP on COCO test-dev with 77M parameters and 410B FLOPs, being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. Code is available at this https URL.
* [[DetNAS](https://arxiv.org/abs/1903.10979)]
    [[pdf](https://arxiv.org/pdf/1903.10979.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1903.10979/)]
    * Title: DetNAS: Backbone Search for Object Detection
    * Year: 26 Mar `2019`
    * Authors: Yukang Chen, Tong Yang, Xiangyu Zhang, Gaofeng Meng, Xinyu Xiao, Jian Sun
    * Abstract: Object detectors are usually equipped with backbone networks designed for image classification. It might be sub-optimal because of the gap between the tasks of image classification and object detection. In this work, we present DetNAS to use Neural Architecture Search (NAS) for the design of better backbones for object detection. It is non-trivial because detection training typically needs ImageNet pre-training while NAS systems require accuracies on the target detection task as supervisory signals. Based on the technique of one-shot supernet, which contains all possible networks in the search space, we propose a framework for backbone search on object detection. We train the supernet under the typical detector training schedule: ImageNet pre-training and detection fine-tuning. Then, the architecture search is performed on the trained supernet, using the detection task as the guidance. This framework makes NAS on backbones very efficient. In experiments, we show the effectiveness of DetNAS on various detectors, for instance, one-stage RetinaNet and the two-stage FPN. We empirically find that networks searched on object detection shows consistent superiority compared to those searched on ImageNet classification. The resulting architecture achieves superior performance than hand-crafted networks on COCO with much less FLOPs complexity.

## Attention Mechanism

* [[Multiple Object Recognition with Visual Attention](https://arxiv.org/abs/1412.7755)]
    [[pdf](https://arxiv.org/pdf/1412.7755.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.7755/)]
    * Title: Multiple Object Recognition with Visual Attention
    * Year: 24 Dec `2014`
    * Authors: Jimmy Ba, Volodymyr Mnih, Koray Kavukcuoglu
    * Abstract: We present an attention-based model for recognizing multiple objects in images. The proposed model is a deep recurrent neural network trained with reinforcement learning to attend to the most relevant regions of the input image. We show that the model learns to both localize and recognize multiple objects despite being given only class labels during training. We evaluate the model on the challenging task of transcribing house number sequences from Google Street View images and show that it is both more accurate than the state-of-the-art convolutional networks and uses fewer parameters and less computation.
* [[Active Object Localization with Deep Reinforcement Learning](https://arxiv.org/abs/1511.06015)]
    [[pdf](https://arxiv.org/pdf/1511.06015.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06015/)]
    * Title: Active Object Localization with Deep Reinforcement Learning
    * Year: 18 Nov `2015`
    * Authors: Juan C. Caicedo, Svetlana Lazebnik
    * Abstract: We present an active detection model for localizing objects in scenes. The model is class-specific and allows an agent to focus attention on candidate regions for identifying the correct location of a target object. This agent learns to deform a bounding box using simple transformation actions, with the goal of determining the most specific location of target objects following top-down reasoning. The proposed localization agent is trained using deep reinforcement learning, and evaluated on the Pascal VOC 2007 dataset. We show that agents guided by the proposed model are able to localize a single instance of an object after analyzing only between 11 and 25 regions in an image, and obtain the best detection results among systems that do not use object proposals for object localization.
* [[AttentionNet](https://arxiv.org/abs/1506.07704)]
    [[pdf](https://arxiv.org/pdf/1506.07704.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1506.07704/)]
    * Title: AttentionNet: Aggregating Weak Directions for Accurate Object Detection
    * Year: 25 Jun `2015`
    * Authors: Donggeun Yoo, Sunggyun Park, Joon-Young Lee, Anthony S. Paek, In So Kweon
    * Abstract: We present a novel detection method using a deep convolutional neural network (CNN), named AttentionNet. We cast an object detection problem as an iterative classification problem, which is the most suitable form of a CNN. AttentionNet provides quantized weak directions pointing a target object and the ensemble of iterative predictions from AttentionNet converges to an accurate object boundary box. Since AttentionNet is a unified network for object detection, it detects objects without any separated models from the object proposal to the post bounding-box regression. We evaluate AttentionNet by a human detection task and achieve the state-of-the-art performance of 65% (AP) on PASCAL VOC 2007/2012 with an 8-layered architecture only.

## Transformer Architectures Applied to Detection

* [[DETR](https://arxiv.org/abs/2005.12872)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2005.12872.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2005.12872/)]
    * Title: End-to-End Object Detection with Transformers
    * Year: 26 May `2020`
    * Authors: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
    * Abstract: We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at this https URL.
    * Comments:
        * > Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019, Wang et al., 2020). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. (ViT, 2020)
        * > DETR utilizes the Transformer decoder to model object detection as an end-to-end dictionary lookup problem with learnable queries, successfully removing the need for handcrafted processes such as NMS. (PVT, )
* [[Deformable DETR](https://arxiv.org/abs/2010.04159)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2010.04159.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2010.04159/)]
    * Title: Deformable DETR: Deformable Transformers for End-to-End Object Detection
    * Year: 08 Oct `2020`
    * Authors: Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai
    * Abstract: DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach. Code is released at this https URL.
    * Comments:
        * > (2021, PVT) Based on DETR, deformable DETR [64] further introduces a deformable attention layer to focus on a sparse set of contextual elements which obtains fast convergence and better performance.

## Weekly-Supervised Learning

* [Is object localization for free? - Weakly-supervised learning with convolutional neural networks](https://ieeexplore.ieee.org/document/7298668)
    * Title: Is object localization for free? - Weakly-supervised learning with convolutional neural networks
