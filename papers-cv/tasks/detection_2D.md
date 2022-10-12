# [Papers][Vision] 2D Object Detection

count: 88

## Unclassified

* [Multi-Stage Features](https://arxiv.org/abs/1212.0142)
    * Title: Pedestrian Detection with Unsupervised Multi-Stage Feature Learning
    * Year: 01 Dec `2012`
    * Author: Pierre Sermanet
    * Abstract: Pedestrian detection is a problem of considerable practical interest. Adding to the list of successful applications of deep learning methods to vision, we report state-of-the-art and competitive results on all major pedestrian datasets with a convolutional network model. The model uses a few new twists, such as multi-stage features, connections that skip layers to integrate global shape information with local distinctive motif information, and an unsupervised method based on convolutional sparse coding to pre-train the filters at each stage.
* [MultiGrasp](https://arxiv.org/abs/1412.3128)
    * Title: Real-Time Grasp Detection Using Convolutional Neural Networks
    * Year: 09 Dec `2014`
    * Authors: Joseph Redmon, Anelia Angelova
    * Abstract: We present an accurate, real-time approach to robotic grasp detection based on convolutional neural networks. Our network performs single-stage regression to graspable bounding boxes without using standard sliding window or region proposal techniques. The model outperforms state-of-the-art approaches by 14 percentage points and runs at 13 frames per second on a GPU. Our network can simultaneously perform classification so that in a single step it recognizes the object and finds a good grasp rectangle. A modification to this model predicts multiple grasps per object by using a locally constrained prediction mechanism. The locally constrained model performs significantly better, especially on objects that can be grasped in a variety of ways.
<!-- * [DPM](https://arxiv.org/abs/1409.5403) -->
<!-- * Title: Deformable Part Models are Convolutional Neural Networks -->
<!-- * Year: 18 Sep `2014` -->
<!-- * Author: Ross Girshick -->
<!-- * Abstract: Deformable part models (DPMs) and convolutional neural networks (CNNs) are two widely used tools for visual recognition. They are typically viewed as distinct approaches: DPMs are graphical models (Markov random fields), while CNNs are "black-box" non-linear classifiers. In this paper, we show that a DPM can be formulated as a CNN, thus providing a novel synthesis of the two ideas. Our construction involves unrolling the DPM inference algorithm and mapping each step to an equivalent (and at times novel) CNN layer. From this perspective, it becomes natural to replace the standard image features used in DPM with a learned feature extractor. We call the resulting model DeepPyramid DPM and experimentally validate it on PASCAL VOC. DeepPyramid DPM significantly outperforms DPMs based on histograms of oriented gradients features (HOG) and slightly outperforms a comparable version of the recently introduced R-CNN detection system, while running an order of magnitude faster. -->
* (07 May 2015) [Multi-Region CNN](https://arxiv.org/abs/1505.01749) (Object detection via a multi-region & semantic segmentation-aware CNN model)
* (15 Oct 2015) [DeepProposal](https://arxiv.org/abs/1510.04445) (DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers)
* (07 Jan 2019) [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)
* (16 Apr 2019) [CenterNet](https://arxiv.org/abs/1904.07850) (Objects as Points)
* (03 Jun 2020) [DetectoRS](https://arxiv.org/abs/2006.02334): Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolutions
* (23 Mar 2021) [Robust and Accurate Object Detection via Adversarial Learning](https://arxiv.org/abs/2103.13886)
* [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189)
    * Title: An Analysis of Scale Invariance in Object Detection - SNIP
    * Year: 22 Nov `2017`
    * Author: Bharat Singh
    * Abstract: An analysis of different techniques for recognizing and detecting objects under extreme scale variation is presented. Scale specific and scale invariant design of detectors are compared by training them with different configurations of input data. By evaluating the performance of different network architectures for classifying small objects on ImageNet, we show that CNNs are not robust to changes in scale. Based on this analysis, we propose to train and test detectors on the same scales of an image-pyramid. Since small and large objects are difficult to recognize at smaller and larger scales respectively, we present a novel training scheme called Scale Normalization for Image Pyramids (SNIP) which selectively back-propagates the gradients of object instances of different sizes as a function of the image scale. On the COCO dataset, our single model performance is 45.7% and an ensemble of 3 networks obtains an mAP of 48.3%. We use off-the-shelf ImageNet-1000 pre-trained models and only train with bounding box supervision. Our submission won the Best Student Entry in the COCO 2017 challenge. Code will be made available at [this http URL](http://bit.ly/2yXVg4c).
* [SNIPER: Efficient Multi-Scale Training](https://arxiv.org/abs/1805.09300)
    * Title: SNIPER: Efficient Multi-Scale Training
    * Year: 23 May `2018`
    * Author: Bharat Singh
    * Abstract: We present SNIPER, an algorithm for performing efficient multi-scale training in instance level visual recognition tasks. Instead of processing every pixel in an image pyramid, SNIPER processes context regions around ground-truth instances (referred to as chips) at the appropriate scale. For background sampling, these context-regions are generated using proposals extracted from a region proposal network trained with a short learning schedule. Hence, the number of chips generated per image during training adaptively changes based on the scene complexity. SNIPER only processes 30% more pixels compared to the commonly used single scale training at 800x1333 pixels on the COCO dataset. But, it also observes samples from extreme resolutions of the image pyramid, like 1400x2000 pixels. As SNIPER operates on resampled low resolution chips (512x512 pixels), it can have a batch size as large as 20 on a single GPU even with a ResNet-101 backbone. Therefore it can benefit from batch-normalization during training without the need for synchronizing batch-normalization statistics across GPUs. SNIPER brings training of instance level recognition tasks like object detection closer to the protocol for image classification and suggests that the commonly accepted guideline that it is important to train on high resolution images for instance level visual recognition tasks might not be correct. Our implementation based on Faster-RCNN with a ResNet-101 backbone obtains an mAP of 47.6% on the COCO dataset for bounding box detection and can process 5 images per second during inference with a single GPU. Code is available at [this https URL](https://github.com/MahyarNajibi/SNIPER/).
* [MobileDets](https://arxiv.org/abs/2004.14525)
    * Title: MobileDets: Searching for Object Detection Architectures for Mobile Accelerators
    * Year: 30 Apr 2020
    * Authors: Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin, Gabriel Bender, Yongzhe Wang, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh, Bo Chen
    * Abstract: Inverted bottleneck layers, which are built upon depthwise convolutions, have been the predominant building blocks in state-of-the-art object detection models on mobile devices. In this work, we investigate the optimality of this design pattern over a broad range of mobile accelerators by revisiting the usefulness of regular convolutions. We discover that regular convolutions are a potent component to boost the latency-accuracy trade-off for object detection on accelerators, provided that they are placed strategically in the network via neural architecture search. By incorporating regular convolutions in the search space and directly optimizing the network architectures for object detection, we obtain a family of object detection models, MobileDets, that achieve state-of-the-art results across mobile accelerators. On the COCO object detection task, MobileDets outperform MobileNetV3+SSDLite by 1.7 mAP at comparable mobile CPU inference latencies. MobileDets also outperform MobileNetV2+SSDLite by 1.9 mAP on mobile CPUs, 3.7 mAP on Google EdgeTPU, 3.4 mAP on Qualcomm Hexagon DSP and 2.7 mAP on Nvidia Jetson GPU without increasing latency. Moreover, MobileDets are comparable with the state-of-the-art MnasFPN on mobile CPUs even without using the feature pyramid, and achieve better mAP scores on both EdgeTPUs and DSPs with up to 2x speedup. Code and models are available in the TensorFlow Object Detection API: [this https URL](https://github.com/tensorflow/models/tree/master/research/object_detection).
* [[Object detectors emerge in Deep Scene CNNs](https://arxiv.org/abs/1412.6856)]
    [[pdf](https://arxiv.org/pdf/1412.6856.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.6856/)]
    * Title: Object detectors emerge in Deep Scene CNNs
    * Year: 22 Dec `2014`
    * Authors: Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba
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

* [What makes for effective detection proposals?](https://arxiv.org/abs/1502.05082)
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

## ----------------------------------------------------------------------------------------------------
## Two-Stage Detectors
## ----------------------------------------------------------------------------------------------------

### unclassified

* [MultiBox](https://arxiv.org/abs/1312.2249)
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
    * Abstract: Current high-quality object detection approaches use the scheme of salience-based object proposal methods followed by post-classification using deep convolutional features. This spurred recent research in improving object proposal methods. However, domain agnostic proposal generation has the principal drawback that the proposals come unranked or with very weak ranking, making it hard to trade-off quality for running time. This raises the more fundamental question of whether high-quality proposal generation requires careful engineering or can be derived just from data alone. We demonstrate that learning-based proposal methods can effectively match the performance of hand-engineered methods while allowing for very efficient runtime-quality trade-offs. Using the multi-scale convolutional MultiBox (MSC-MultiBox) approach, we substantially advance the state-of-the-art on the ILSVRC 2014 detection challenge data set, with 0.5 mAP for a single model and 0.52 mAP for an ensemble of two models. MSC-Multibox significantly improves the proposal quality over its predecessor MultiBox~method: AP increases from 0.42 to 0.53 for the ILSVRC detection challenge. Finally, we demonstrate improved bounding-box recall compared to Multiscale Combinatorial Grouping with less proposals on the Microsoft-COCO data set.
* [DeepBox](https://arxiv.org/abs/1505.02146)
    * Title: DeepBox: Learning Objectness with Convolutional Networks
    * Year: 08 May `2015`
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
    * Abstract: Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012---achieving a mAP of 53.3%. Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also compare R-CNN to OverFeat, a recently proposed sliding-window detector based on a similar CNN architecture. We find that R-CNN outperforms OverFeat by a large margin on the 200-class ILSVRC2013 detection dataset. Source code for the complete system is available at [this http URL](http://www.cs.berkeley.edu/~rbg/rcnn).
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
    * Abstract: This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3x faster, tests 10x faster, and is more accurate. Fast R-CNN is implemented in Python and C++ (using Caffe) and is available under the open-source MIT License at [this https URL](https://github.com/rbgirshick/fast-rcnn).
    * Comments:
        * > (2016, FPN) Recent and more accurate detection methods like Fast R-CNN [11] and Faster R-CNN [29] advocate using features computed from a single scale, because it offers a good trade-off between accuracy and speed.
* [Faster R-CNN/RPN](https://arxiv.org/abs/1506.01497)
    * Title: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
    * Year: 04 Jun `2015`
    * Author: Shaoqing Ren
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
    * Abstract: This paper presents a method for recognizing scene categories based on approximate global geometric correspondence. This technique works by partitioning the image into increasingly fine sub-regions and computing histograms of local features found inside each sub-region. The resulting "spatial pyramid" is a simple and computationally efficient extension of an orderless bag-of-features image representation, and it shows significantly improved performance on challenging scene categorization tasks. Specifically, our proposed method exceeds the state of the art on the Caltech-101 database and achieves high accuracy on a large database of fifteen natural scene categories. The spatial pyramid framework also offers insights into the success of several recently proposed image descriptions, including Torralba’s "gist" and Lowe’s SIFT descriptors.
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

### Further Improvements of R-CNN

* [R-FCN](https://arxiv.org/abs/1605.06409) <!-- printed -->
    * Title: R-FCN: Object Detection via Region-based Fully Convolutional Networks
    * Year: 20 May `2016`
    * Author: Jifeng Dai
    * Abstract: We present region-based, fully convolutional networks for accurate and efficient object detection. In contrast to previous region-based detectors such as Fast/Faster R-CNN that apply a costly per-region subnetwork hundreds of times, our region-based detector is fully convolutional with almost all computation shared on the entire image. To achieve this goal, we propose position-sensitive score maps to address a dilemma between translation-invariance in image classification and translation-variance in object detection. Our method can thus naturally adopt fully convolutional image classifier backbones, such as the latest Residual Networks (ResNets), for object detection. We show competitive results on the PASCAL VOC datasets (e.g., 83.6% mAP on the 2007 set) with the 101-layer ResNet. Meanwhile, our result is achieved at a test-time speed of 170ms per image, 2.5-20x faster than the Faster R-CNN counterpart. Code is made publicly available at: [this https URL](https://github.com/daijifeng001/r-fcn).
* [Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection](https://arxiv.org/abs/1604.04693)
    * Title: Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection
    * Year: 16 Apr `2016`
    * Author: Yu Xiang
    * Abstract: In CNN-based object detection methods, region proposal becomes a bottleneck when objects exhibit significant scale variation, occlusion or truncation. In addition, these methods mainly focus on 2D object detection and cannot estimate detailed properties of objects. In this paper, we propose subcategory-aware CNNs for object detection. We introduce a novel region proposal network that uses subcategory information to guide the proposal generating process, and a new detection network for joint detection and subcategory classification. By using subcategories related to object pose, we achieve state-of-the-art performance on both detection and pose estimation on commonly used benchmarks.
* [[Mask R-CNN](https://arxiv.org/abs/1703.06870)]
    * Title: Mask R-CNN
    * Year: 20 Mar `2017`
    * Author: Kaiming He
    * Abstract: We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code has been made available at: [this https URL](https://github.com/facebookresearch/Detectron).
* [[Cascade R-CNN](https://arxiv.org/abs/1712.00726)]
    * Title: Cascade R-CNN: Delving into High Quality Object Detection
    * Year: 03 Dec `2017`
    * Author: Zhaowei Cai
    * Abstract: In object detection, an intersection over union (IoU) threshold is required to define positives and negatives. An object detector, trained with low IoU threshold, e.g. 0.5, usually produces noisy detections. However, detection performance tends to degrade with increasing the IoU thresholds. Two main factors are responsible for this: 1) overfitting during training, due to exponentially vanishing positive samples, and 2) inference-time mismatch between the IoUs for which the detector is optimal and those of the input hypotheses. A multi-stage object detection architecture, the Cascade R-CNN, is proposed to address these problems. It consists of a sequence of detectors trained with increasing IoU thresholds, to be sequentially more selective against close false positives. The detectors are trained stage by stage, leveraging the observation that the output of a detector is a good distribution for training the next higher quality detector. The resampling of progressively improved hypotheses guarantees that all detectors have a positive set of examples of equivalent size, reducing the overfitting problem. The same cascade procedure is applied at inference, enabling a closer match between the hypotheses and the detector quality of each stage. A simple implementation of the Cascade R-CNN is shown to surpass all single-model object detectors on the challenging COCO dataset. Experiments also show that the Cascade R-CNN is widely applicable across detector architectures, achieving consistent gains independently of the baseline detector strength. The code will be made available at [this https URL](https://github.com/zhaoweicai/cascade-rcnn).
* [Sparse R-CNN](https://arxiv.org/abs/2011.12450)
    * Title: Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
    * Year: 25 Nov `2020`
    * Authors: Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, Ping Luo
    * Abstract: We present Sparse R-CNN, a purely sparse method for object detection in images. Existing works on object detection heavily rely on dense object candidates, such as $k$ anchor boxes pre-defined on all grids of image feature map of size $H \times W$. In our method, however, a fixed sparse set of learned object proposals, total length of $N$, are provided to object recognition head to perform classification and location. By eliminating $HWk$ (up to hundreds of thousands) hand-designed object candidates to $N$ (e.g. 100) learnable proposals, Sparse R-CNN completely avoids all efforts related to object candidates design and many-to-one label assignment. More importantly, final predictions are directly output without non-maximum suppression post-procedure. Sparse R-CNN demonstrates accuracy, run-time and training convergence performance on par with the well-established detector baselines on the challenging COCO dataset, e.g., achieving 45.0 AP in standard $3 \times$ training schedule and running at 22 fps using ResNet-50 FPN model. We hope our work could inspire re-thinking the convention of dense prior in object detectors. The code is available at: [this https URL](https://github.com/PeizeSun/SparseR-CNN).
* [Feature Selective Networks](https://arxiv.org/abs/1711.08879)
    * Title: Feature Selective Networks for Object Detection
    * Year: 24 Nov `2017`
    * Author: Yao Zhai
    * Abstract: Objects for detection usually have distinct characteristics in different sub-regions and different aspect ratios. However, in prevalent two-stage object detection methods, Region-of-Interest (RoI) features are extracted by RoI pooling with little emphasis on these translation-variant feature components. We present feature selective networks to reform the feature representations of RoIs by exploiting their disparities among sub-regions and aspect ratios. Our network produces the sub-region attention bank and aspect ratio attention bank for the whole image. The RoI-based sub-region attention map and aspect ratio attention map are selectively pooled from the banks, and then used to refine the original RoI features for RoI classification. Equipped with a light-weight detection subnetwork, our network gets a consistent boost in detection performance based on general ConvNet backbones (ResNet-101, GoogLeNet and VGG-16). Without bells and whistles, our detectors equipped with ResNet-101 achieve more than 3% mAP improvement compared to counterparts on PASCAL VOC 2007, PASCAL VOC 2012 and MS COCO datasets.
* [Dynamic R-CNN](https://arxiv.org/abs/2004.06002)
    * Title: Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training
    * Year: 13 Apr `2020`
    * Authors: Hongkai Zhang, Hong Chang, Bingpeng Ma, Naiyan Wang, Xilin Chen
    * Abstract: Although two-stage object detectors have continuously advanced the state-of-the-art performance in recent years, the training process itself is far from crystal. In this work, we first point out the inconsistency problem between the fixed network settings and the dynamic training procedure, which greatly affects the performance. For example, the fixed label assignment strategy and regression loss function cannot fit the distribution change of proposals and thus are harmful to training high quality detectors. Consequently, we propose Dynamic R-CNN to adjust the label assignment criteria (IoU threshold) and the shape of regression loss function (parameters of SmoothL1 Loss) automatically based on the statistics of proposals during training. This dynamic design makes better use of the training samples and pushes the detector to fit more high quality samples. Specifically, our method improves upon ResNet-50-FPN baseline with 1.9% AP and 5.5% AP90 on the MS COCO dataset with no extra overhead. Codes and models are available at [this https URL](https://github.com/hkzhang95/DynamicRCNN).

## ----------------------------------------------------------------------------------------------------
## Single-Stage Detectors
## ----------------------------------------------------------------------------------------------------

### unclassified

* [PolarMask](https://arxiv.org/abs/1909.13226)
    * Title: PolarMask: Single Shot Instance Segmentation with Polar Representation
    * Year: 29 Sep `2019`
    * Authors: Enze Xie, Peize Sun, Xiaoge Song, Wenhai Wang, Ding Liang, Chunhua Shen, Ping Luo
    * Abstract: In this paper, we introduce an anchor-box free and single shot instance segmentation method, which is conceptually simple, fully convolutional and can be used as a mask prediction module for instance segmentation, by easily embedding it into most off-the-shelf detection methods. Our method, termed PolarMask, formulates the instance segmentation problem as instance center classification and dense distance regression in a polar coordinate. Moreover, we propose two effective approaches to deal with sampling high-quality center examples and optimization for dense distance regression, respectively, which can significantly improve the performance and simplify the training process. Without any bells and whistles, PolarMask achieves 32.9% in mask mAP with single-model and single-scale training/testing on challenging COCO dataset. For the first time, we demonstrate a much simpler and flexible instance segmentation framework achieving competitive accuracy. We hope that the proposed PolarMask framework can serve as a fundamental and strong baseline for single shot instance segmentation tasks. Code is available at: [this http URL](http://github.com/xieenze/PolarMask).
* [OneNet](https://arxiv.org/abs/2012.05780)
    * Title: What Makes for End-to-End Object Detection
    * Year: 10 Dec `2020`
    * Authors: Peize Sun, Yi Jiang, Enze Xie, Wenqi Shao, Zehuan Yuan, Changhu Wang, Ping Luo
    * Abstract: Object detection has recently achieved a breakthrough for removing the last one non-differentiable component in the pipeline, Non-Maximum Suppression (NMS), and building up an end-to-end system. However, what makes for its one-to-one prediction has not been well understood. In this paper, we first point out that one-to-one positive sample assignment is the key factor, while, one-to-many assignment in previous detectors causes redundant predictions in inference. Second, we surprisingly find that even training with one-to-one assignment, previous detectors still produce redundant predictions. We identify that classification cost in matching cost is the main ingredient: (1) previous detectors only consider location cost, (2) by additionally introducing classification cost, previous detectors immediately produce one-to-one prediction during inference. We introduce the concept of score gap to explore the effect of matching cost. Classification cost enlarges the score gap by choosing positive samples as those of highest score in the training iteration and reducing noisy positive samples brought by only location cost. Finally, we demonstrate the advantages of end-to-end object detection on crowded scenes. The code is available at: [this https URL](https://github.com/PeizeSun/OneNet).
* [TDM](https://arxiv.org/abs/1612.06851)
    * Title: Beyond Skip Connections: Top-Down Modulation for Object Detection
    * Year: 20 Dec `2016`
    * Author: Abhinav Shrivastava
    * Abstract: In recent years, we have seen tremendous progress in the field of object detection. Most of the recent improvements have been achieved by targeting deeper feedforward networks. However, many hard object categories such as bottle, remote, etc. require representation of fine details and not just coarse, semantic representations. But most of these fine details are lost in the early convolutional layers. What we need is a way to incorporate finer details from lower layers into the detection architecture. Skip connections have been proposed to combine high-level and low-level features, but we argue that selecting the right features from low-level requires top-down contextual information. Inspired by the human visual pathway, in this paper we propose top-down modulations as a way to incorporate fine details into the detection framework. Our approach supplements the standard bottom-up, feedforward ConvNet with a top-down modulation (TDM) network, connected using lateral connections. These connections are responsible for the modulation of lower layer filters, and the top-down network handles the selection and integration of contextual information and low-level features. The proposed TDM architecture provides a significant boost on the COCO testdev benchmark, achieving 28.6 AP for VGG16, 35.2 AP for ResNet101, and 37.3 for InceptionResNetv2 network, without any bells and whistles (e.g., multi-scale, iterative box refinement, etc.).
* FCOS: Fully Convolutional One-Stage Object Detection

### SSD and its Variants

* [SSD](https://arxiv.org/abs/1512.02325) <!-- printed -->
    * Title: SSD: Single Shot MultiBox Detector
    * Year: 08 Dec `2015`
    * Author: Wei Liu
    * Abstract: We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For 300x300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for 500x500 input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model. Code is available at [this https URL](https://github.com/weiliu89/caffe/tree/ssd).
    * Comments:
        * > (2016, FPN) The Single Shot Detector (SSD) [22] is one of the first attempts at using a ConvNet’s pyramidal feature hierarchy as if it were a featurized image pyramid (Fig. 1(c)).
        * > (2016, FPN) SSD [22] and MS-CNN [3] predict objects at multiple layers of the feature hierarchy without combining features or scores.
* [DSSD](https://arxiv.org/abs/1701.06659)
    * Title: DSSD : Deconvolutional Single Shot Detector
    * Year: 23 Jan `2017`
    * Author: Cheng-Yang Fu
    * Abstract: The main contribution of this paper is an approach for introducing additional context into state-of-the-art general object detection. To achieve this we first combine a state-of-the-art classifier (Residual-101[14]) with a fast detection framework (SSD[18]). We then augment SSD+Residual-101 with deconvolution layers to introduce additional large-scale context in object detection and improve accuracy, especially for small objects, calling our resulting system DSSD for deconvolutional single shot detector. While these two contributions are easily described at a high-level, a naive implementation does not succeed. Instead we show that carefully adding additional stages of learned transformations, specifically a module for feed-forward connections in deconvolution and a new output module, enables this new approach and forms a potential way forward for further detection research. Results are shown on both PASCAL VOC and COCO detection. Our DSSD with 513x513 input achieves 81.5% mAP on VOC2007 test, 80.0% mAP on VOC2012 test, and 33.2% mAP on COCO, outperforming a state-of-the-art method R-FCN[3] on each dataset.
* [FSSD](https://arxiv.org/abs/1712.00960)
    * Title: FSSD: Feature Fusion Single Shot Multibox Detector
    * Year: 04 Dec `2017`
    * Author: Zuoxin Li
    * Abstract: SSD (Single Shot Multibox Detector) is one of the best object detection algorithms with both high accuracy and fast speed. However, SSD's feature pyramid detection method makes it hard to fuse the features from different scales. In this paper, we proposed FSSD (Feature Fusion Single Shot Multibox Detector), an enhanced SSD with a novel and lightweight feature fusion module which can improve the performance significantly over SSD with just a little speed drop. In the feature fusion module, features from different layers with different scales are concatenated together, followed by some down-sampling blocks to generate new feature pyramid, which will be fed to multibox detectors to predict the final detection results. On the Pascal VOC 2007 test, our network can achieve 82.7 mAP (mean average precision) at the speed of 65.8 FPS (frame per second) with the input size 300x300 using a single Nvidia 1080Ti GPU. In addition, our result on COCO is also better than the conventional SSD with a large margin. Our FSSD outperforms a lot of state-of-the-art object detection algorithms in both aspects of accuracy and speed. Code is available at [this https URL](https://github.com/lzx1413/CAFFE_SSD/tree/fssd).

### YOLO Series

* [[YOLOv1](https://arxiv.org/abs/1506.02640)] <!-- printed -->
    * Title: You Only Look Once: Unified, Real-Time Object Detection
    * Year: 08 Jun `2015`
    * Authors: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
    * Abstract: We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. Compared to state-of-the-art detection systems, YOLO makes more localization errors but is far less likely to predict false detections where nothing exists. Finally, YOLO learns very general representations of objects. It outperforms all other detection methods, including DPM and R-CNN, by a wide margin when generalizing from natural images to artwork on both the Picasso Dataset and the People-Art Dataset.
* [[YOLOv2](https://arxiv.org/abs/1612.08242)] <!-- printed -->
    * Title: YOLO9000: Better, Faster, Stronger
    * Year: 25 Dec `2016`
    * Authors: Joseph Redmon, Ali Farhadi
    * Abstract: We introduce YOLO9000, a state-of-the-art, real-time object detection system that can detect over 9000 object categories. First we propose various improvements to the YOLO detection method, both novel and drawn from prior work. The improved model, YOLOv2, is state-of-the-art on standard detection tasks like PASCAL VOC and COCO. At 67 FPS, YOLOv2 gets 76.8 mAP on VOC 2007. At 40 FPS, YOLOv2 gets 78.6 mAP, outperforming state-of-the-art methods like Faster RCNN with ResNet and SSD while still running significantly faster. Finally we propose a method to jointly train on object detection and classification. Using this method we train YOLO9000 simultaneously on the COCO detection dataset and the ImageNet classification dataset. Our joint training allows YOLO9000 to predict detections for object classes that don't have labelled detection data. We validate our approach on the ImageNet detection task. YOLO9000 gets 19.7 mAP on the ImageNet detection validation set despite only having detection data for 44 of the 200 classes. On the 156 classes not in COCO, YOLO9000 gets 16.0 mAP. But YOLO can detect more than just 200 classes; it predicts detections for more than 9000 different object categories. And it still runs in real-time.
* [[YOLOv3](https://arxiv.org/abs/1804.02767)] <!-- printed -->
    * Title: YOLOv3: An Incremental Improvement
    * Year: 08 Apr `2018`
    * Authors: Joseph Redmon, Ali Farhadi
    * Abstract: We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5 mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster. As always, all the code is online at [this https URL](https://pjreddie.com/yolo/).
* [[YOLOv4](https://arxiv.org/abs/2004.10934)] <!-- printed -->
    * Title: YOLOv4: Optimal Speed and Accuracy of Object Detection
    * Year: 23 Apr `2020`
    * Authors: Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
    * Abstract: There are a huge number of features which are said to improve Convolutional Neural Network (CNN) accuracy. Practical testing of combinations of such features on large datasets, and theoretical justification of the result, is required. Some features operate on certain models exclusively and for certain problems exclusively, or only for small-scale datasets; while some features, such as batch-normalization and residual-connections, are applicable to the majority of models, tasks, and datasets. We assume that such universal features include Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT) and Mish-activation. We use new features: WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, CmBN, DropBlock regularization, and CIoU loss, and combine some of them to achieve state-of-the-art results: 43.5% AP (65.7% AP50) for the MS COCO dataset at a realtime speed of ~65 FPS on Tesla V100. Source code is at [this https URL](https://github.com/AlexeyAB/darknet).
* [Scaled-YOLOv4](https://arxiv.org/abs/2011.08036)
    * Title: Scaled-YOLOv4: Scaling Cross Stage Partial Network
    * Year: 16 Nov `2020`
    * Authors: 
    * Abstract: 
* [YOLOF](https://arxiv.org/abs/2103.09460)
    * Title: You Only Look One-level Feature
    * Year: 17 Mar `2021`
    * Authors: 
    * Abstract: 
* [YOLOR](https://arxiv.org/abs/2105.04206)
    * Title: You Only Learn One Representation
    * Year: 10 May `2021`
    * Authors: 
    * Abstract: 
* [YOLOS](https://arxiv.org/abs/2106.00666)
    * Title: You Only Look at One Sequence
    * Year: 01 Jun `2021`
    * Authors: 
    * Abstract: 
* [YOLOX](https://arxiv.org/abs/2107.08430)
    * Title: Exceeding YOLO Series in 2021
    * Year: 18 Jul `2021`
    * Authors: 
    * Abstract: 

### Methods using multiple layers (2016, FPN)

* [Inside-Outside Net (ION)](https://arxiv.org/abs/1512.04143) <!-- printed -->
    * Title: Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
    * Year: 14 Dec `2015`
    * Author: Sean Bell
    * Abstract: It is well known that contextual and multi-scale representations are important for accurate visual recognition. In this paper we present the Inside-Outside Net (ION), an object detector that exploits information both inside and outside the region of interest. Contextual information outside the region of interest is integrated using spatial recurrent neural networks. Inside, we use skip pooling to extract information at multiple scales and levels of abstraction. Through extensive experiments we evaluate the design space and provide readers with an overview of what tricks of the trade are important. ION improves state-of-the-art on PASCAL VOC 2012 object detection from 73.9% to 76.4% mAP. On the new and more challenging MS COCO dataset, we improve state-of-art-the from 19.7% to 33.1% mAP. In the 2015 MS COCO Detection Challenge, our ION model won the Best Student Entry and finished 3rd place overall. As intuition suggests, our detection results provide strong evidence that context and multi-scale representations improve small object detection.
    * Comments:
        * > (2016, FPN) Several other approaches (HyperNet [18], ParseNet [23], and ION [2]) concatenate features of multiple layers before computing predictions, which is equivalent to summing transformed features.
* [HyperNet](https://arxiv.org/abs/1604.00600)
    * Title: HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection
    * Year: 03 Apr `2016`
    * Author: Tao Kong
    * Abstract: Almost all of the current top-performing object detection networks employ region proposals to guide the search for object instances. State-of-the-art region proposal methods usually need several thousand proposals to get high recall, thus hurting the detection efficiency. Although the latest Region Proposal Network method gets promising detection accuracy with several hundred proposals, it still struggles in small-size object detection and precise localization (e.g., large IoU thresholds), mainly due to the coarseness of its feature maps. In this paper, we present a deep hierarchical network, namely HyperNet, for handling region proposal generation and object detection jointly. Our HyperNet is primarily based on an elaborately designed Hyper Feature which aggregates hierarchical feature maps first and then compresses them into a uniform space. The Hyper Features well incorporate deep but highly semantic, intermediate but really complementary, and shallow but naturally high-resolution features of the image, thus enabling us to construct HyperNet by sharing them both in generating proposals and detecting objects via an end-to-end joint training strategy. For the deep VGG16 model, our method achieves completely leading recall and state-of-the-art object detection accuracy on PASCAL VOC 2007 and 2012 using only 100 proposals per image. It runs with a speed of 5 fps (including all steps) on a GPU, thus having the potential for real-time processing.
    * Comments:
        * > (2016, FPN) Several other approaches (HyperNet [18], ParseNet [23], and ION [2]) concatenate features of multiple layers before computing predictions, which is equivalent to summing transformed features.
* [[MS-CNN](https://arxiv.org/abs/1607.07155)]
    * Title: A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection
    * Year: 25 Jul `2016`
    * Author: Zhaowei Cai
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
        * > In contrast to ‘symmetric’ decoders [47], FPN uses a lightweight decoder (see Fig. 5). (Panoptic FPN, 2019)

### Improvements

* [OHEM](https://arxiv.org/abs/1604.03540)
    * Title: Training Region-based Object Detectors with Online Hard Example Mining
    * Year: 12 Apr `2016`
    * Author: Abhinav Shrivastava
    * Abstract: The field of object detection has made significant advances riding on the wave of region-based ConvNets, but their training procedure still includes many heuristics and hyperparameters that are costly to tune. We present a simple yet surprisingly effective online hard example mining (OHEM) algorithm for training region-based ConvNet detectors. Our motivation is the same as it has always been -- detection datasets contain an overwhelming number of easy examples and a small number of hard examples. Automatic selection of these hard examples can make training more effective and efficient. OHEM is a simple and intuitive algorithm that eliminates several heuristics and hyperparameters in common use. But more importantly, it yields consistent and significant boosts in detection performance on benchmarks like PASCAL VOC 2007 and 2012. Its effectiveness increases as datasets become larger and more difficult, as demonstrated by the results on the MS COCO dataset. Moreover, combined with complementary advances in the field, OHEM leads to state-of-the-art results of 78.9% and 76.3% mAP on PASCAL VOC 2007 and 2012 respectively.
* (29 Aug 2016) [PVANET](https://arxiv.org/abs/1608.08021) (PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection)
* [RON](https://arxiv.org/abs/1707.01691)
    * Title: RON: Reverse Connection with Objectness Prior Networks for Object Detection
    * Year: 06 Jul 2017
    * Author: Tao Kong
    * Abstract: We present RON, an efficient and effective framework for generic object detection. Our motivation is to smartly associate the best of the region-based (e.g., Faster R-CNN) and region-free (e.g., SSD) methodologies. Under fully convolutional architecture, RON mainly focuses on two fundamental problems: (a) multi-scale object localization and (b) negative sample mining. To address (a), we design the reverse connection, which enables the network to detect objects on multi-levels of CNNs. To deal with (b), we propose the objectness prior to significantly reduce the searching space of objects. We optimize the reverse connection, objectness prior and object detector jointly by a multi-task loss function, thus RON can directly predict final detection results from all locations of various feature maps. Extensive experiments on the challenging PASCAL VOC 2007, PASCAL VOC 2012 and MS COCO benchmarks demonstrate the competitive performance of RON. Specifically, with VGG-16 and low resolution 384X384 input size, the network gets 81.3% mAP on PASCAL VOC 2007, 80.7% mAP on PASCAL VOC 2012 datasets. Its superiority increases when datasets become larger and more difficult, as demonstrated by the results on the MS COCO dataset. With 1.5G GPU memory at test phase, the speed of the network is 15 FPS, 3X faster than the Faster R-CNN counterpart.
* (09 Aug 2017) [CoupleNet](https://arxiv.org/abs/1708.02863) (CoupleNet: Coupling Global Structure with Local Parts for Object Detection)
* (03 Aug 2017) [DSOD](https://arxiv.org/abs/1708.01241) (DSOD: Learning Deeply Supervised Object Detectors from Scratch)

### Focal Loss

* [[Focal Loss/RetinaNet](https://arxiv.org/abs/1708.02002)] <!-- printed -->
    * Title: Focal Loss for Dense Object Detection
    * Year: 07 Aug `2017`
    * Authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
    * Abstract: The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors. Code is at: [this https URL](https://github.com/facebookresearch/Detectron).
* [Generalized Focal Loss V1](https://arxiv.org/abs/2006.04388)
    * Title: Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection
    * Year: 08 Jun `2020`
    * Authors: Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang
    * Abstract: One-stage detector basically formulates object detection as dense classification and localization. The classification is usually optimized by Focal Loss and the box location is commonly learned under Dirac delta distribution. A recent trend for one-stage detectors is to introduce an individual prediction branch to estimate the quality of localization, where the predicted quality facilitates the classification to improve detection performance. This paper delves into the representations of the above three fundamental elements: quality estimation, classification and localization. Two problems are discovered in existing practices, including (1) the inconsistent usage of the quality estimation and classification between training and inference and (2) the inflexible Dirac delta distribution for localization when there is ambiguity and uncertainty in complex scenes. To address the problems, we design new representations for these elements. Specifically, we merge the quality estimation into the class prediction vector to form a joint representation of localization quality and classification, and use a vector to represent arbitrary distribution of box locations. The improved representations eliminate the inconsistency risk and accurately depict the flexible distribution in real data, but contain continuous labels, which is beyond the scope of Focal Loss. We then propose Generalized Focal Loss (GFL) that generalizes Focal Loss from its discrete form to the continuous version for successful optimization. On COCO test-dev, GFL achieves 45.0\% AP using ResNet-101 backbone, surpassing state-of-the-art SAPD (43.5\%) and ATSS (43.6\%) with higher or comparable inference speed, under the same backbone and training settings. Notably, our best model can achieve a single-model single-scale AP of 48.2\%, at 10 FPS on a single 2080Ti GPU. Code and models are available at [this https URL](https://github.com/implus/GFocal).
* [Generalized Focal Loss V2](https://arxiv.org/abs/2011.12885)
    * Title: Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection
    * Year: 25 Nov `2020`
    * Authors: Xiang Li, Wenhai Wang, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang
    * Abstract: Localization Quality Estimation (LQE) is crucial and popular in the recent advancement of dense object detectors since it can provide accurate ranking scores that benefit the Non-Maximum Suppression processing and improve detection performance. As a common practice, most existing methods predict LQE scores through vanilla convolutional features shared with object classification or bounding box regression. In this paper, we explore a completely novel and different perspective to perform LQE -- based on the learned distributions of the four parameters of the bounding box. The bounding box distributions are inspired and introduced as "General Distribution" in GFLV1, which describes the uncertainty of the predicted bounding boxes well. Such a property makes the distribution statistics of a bounding box highly correlated to its real localization quality. Specifically, a bounding box distribution with a sharp peak usually corresponds to high localization quality, and vice versa. By leveraging the close correlation between distribution statistics and the real localization quality, we develop a considerably lightweight Distribution-Guided Quality Predictor (DGQP) for reliable LQE based on GFLV1, thus producing GFLV2. To our best knowledge, it is the first attempt in object detection to use a highly relevant, statistical representation to facilitate LQE. Extensive experiments demonstrate the effectiveness of our method. Notably, GFLV2 (ResNet-101) achieves 46.2 AP at 14.6 FPS, surpassing the previous state-of-the-art ATSS baseline (43.6 AP at 14.6 FPS) by absolute 2.6 AP on COCO {\tt test-dev}, without sacrificing the efficiency both in training and inference. Code will be available at [this https URL](https://github.com/implus/GFocalV2).

## Anchor-Free Frameworks

* [DeNet](https://arxiv.org/abs/1703.10295)
    * Title: DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling
    * Year: 30 Mar `2017`
    * Author: Lachlan Tychsen-Smith
    * Abstract: We define the object detection from imagery problem as estimating a very large but extremely sparse bounding box dependent probability distribution. Subsequently we identify a sparse distribution estimation scheme, Directed Sparse Sampling, and employ it in a single end-to-end CNN based detection model. This methodology extends and formalizes previous state-of-the-art detection models with an additional emphasis on high evaluation rates and reduced manual engineering. We introduce two novelties, a corner based region-of-interest estimator and a deconvolution based CNN model. The resulting model is scene adaptive, does not require manually defined reference bounding boxes and produces highly competitive results on MSCOCO, Pascal VOC 2007 and Pascal VOC 2012 with real-time evaluation rates. Further analysis suggests our model performs particularly well when fine-grained object localization is desirable. We argue that this advantage stems from the significantly larger set of available regions-of-interest relative to other methods. Source-code is available from: [this https URL](https://github.com/lachlants/denet).
* [PLN](https://arxiv.org/abs/1706.03646)
    * Title: Point Linking Network for Object Detection
    * Year: 12 Jun `2017`
    * Author: Xinggang Wang
    * Abstract: Object detection is a core problem in computer vision. With the development of deep ConvNets, the performance of object detectors has been dramatically improved. The deep ConvNets based object detectors mainly focus on regressing the coordinates of bounding box, e.g., Faster-R-CNN, YOLO and SSD. Different from these methods that considering bounding box as a whole, we propose a novel object bounding box representation using points and links and implemented using deep ConvNets, termed as Point Linking Network (PLN). Specifically, we regress the corner/center points of bounding-box and their links using a fully convolutional network; then we map the corner points and their links back to multiple bounding boxes; finally an object detection result is obtained by fusing the multiple bounding boxes. PLN is naturally robust to object occlusion and flexible to object scale variation and aspect ratio variation. In the experiments, PLN with the Inception-v2 model achieves state-of-the-art single-model and single-scale results on the PASCAL VOC 2007, the PASCAL VOC 2012 and the COCO detection benchmarks without bells and whistles. The source code will be released.
* [CornerNet](https://arxiv.org/abs/1808.01244)
    * Title: CornerNet: Detecting Objects as Paired Keypoints
    * Year: 03 Aug `2018`
    * Author: Hei Law
    * We propose CornerNet, a new approach to object detection where we detect an object bounding box as a pair of keypoints, the top-left corner and the bottom-right corner, using a single convolution neural network. By detecting objects as paired keypoints, we eliminate the need for designing a set of anchor boxes commonly used in prior single-stage detectors. In addition to our novel formulation, we introduce corner pooling, a new type of pooling layer that helps the network better localize corners. Experiments show that CornerNet achieves a 42.2% AP on MS COCO, outperforming all existing one-stage detectors.
* (23 Jan 2019) [ExtremeNet](https://arxiv.org/abs/1901.08043) (Bottom-up Object Detection by Grouping Extreme and Center Points)
* [FCOS](https://arxiv.org/abs/1904.01355)
    * Title: FCOS: Fully Convolutional One-Stage Object Detection
    * Year: 02 Apr `2019`
    * Authors: Zhi Tian, Chunhua Shen, Hao Chen, Tong He
    * Abstract: We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7% in AP with single-model and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler. For the first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks. Code is available at:Code is available at: [this https URL](https://tinyurl.com/FCOSv1).
* (17 Apr 2019) [CenterNet](https://arxiv.org/abs/1904.08189) (CenterNet: Keypoint Triplets for Object Detection)
* (02 Mar 2019) [FSAF](https://arxiv.org/abs/1903.00621) (Feature Selective Anchor-Free Module for Single-Shot Object Detection)
* (16 Apr 2019) [NAS-FPN](https://arxiv.org/abs/1904.07392) (NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection)
* (26 Mar 2019) [DetNAS](https://arxiv.org/abs/1903.10979) (DetNAS: Backbone Search for Object Detection)

## Neural Architecture Search (EfficientNetV2, 2021)

* [EfficientDet](https://arxiv.org/abs/1911.09070)
    * Title: EfficientDet: Scalable and Efficient Object Detection
    * Year: 20 Nov `2019`
    * Authors: Mingxing Tan, Ruoming Pang, Quoc V. Le
    * Abstract: Model efficiency has become increasingly important in computer vision. In this paper, we systematically study neural network architecture design choices for object detection and propose several key optimizations to improve efficiency. First, we propose a weighted bi-directional feature pyramid network (BiFPN), which allows easy and fast multiscale feature fusion; Second, we propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time. Based on these optimizations and better backbones, we have developed a new family of object detectors, called EfficientDet, which consistently achieve much better efficiency than prior art across a wide spectrum of resource constraints. In particular, with single model and single-scale, our EfficientDet-D7 achieves state-of-the-art 55.1 AP on COCO test-dev with 77M parameters and 410B FLOPs, being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. Code is available at [this https URL](https://github.com/google/automl/tree/master/efficientdet).
* [DetNAS](https://arxiv.org/abs/1903.10979)
    * Title: DetNAS: Backbone Search for Object Detection
    * Year: 26 Mar `2019`
    * Authors: Yukang Chen, Tong Yang, Xiangyu Zhang, Gaofeng Meng, Xinyu Xiao, Jian Sun
    * Abstract: Object detectors are usually equipped with backbone networks designed for image classification. It might be sub-optimal because of the gap between the tasks of image classification and object detection. In this work, we present DetNAS to use Neural Architecture Search (NAS) for the design of better backbones for object detection. It is non-trivial because detection training typically needs ImageNet pre-training while NAS systems require accuracies on the target detection task as supervisory signals. Based on the technique of one-shot supernet, which contains all possible networks in the search space, we propose a framework for backbone search on object detection. We train the supernet under the typical detector training schedule: ImageNet pre-training and detection fine-tuning. Then, the architecture search is performed on the trained supernet, using the detection task as the guidance. This framework makes NAS on backbones very efficient. In experiments, we show the effectiveness of DetNAS on various detectors, for instance, one-stage RetinaNet and the two-stage FPN. We empirically find that networks searched on object detection shows consistent superiority compared to those searched on ImageNet classification. The resulting architecture achieves superior performance than hand-crafted networks on COCO with much less FLOPs complexity.

## Attention Mechanism

* [Multiple Object Recognition with Visual Attention](https://arxiv.org/abs/1412.7755)
    [[vanity](https://www.arxiv-vanity.com/papers/1412.7755/)]
    * Title: Multiple Object Recognition with Visual Attention
    * Authors: Jimmy Ba, Volodymyr Mnih, Koray Kavukcuoglu
* [Active Object Localization with Deep Reinforcement Learning](https://arxiv.org/abs/1511.06015)
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06015/)]
    * Title: Active Object Localization with Deep Reinforcement Learning
    * Authors: Juan C. Caicedo, Svetlana Lazebnik
* [AttentionNet](https://arxiv.org/abs/1506.07704)
    [[vanity](https://www.arxiv-vanity.com/papers/1506.07704/)]
    * Title: AttentionNet: Aggregating Weak Directions for Accurate Object Detection
    * Year: 25 Jun `2015`
    * Authors: Donggeun Yoo, Sunggyun Park, Joon-Young Lee, Anthony S. Paek, In So Kweon

## Transformer Architectures Applied to Detection

* [[DETR](https://arxiv.org/abs/2005.12872)] <!-- printed -->
    * Title: End-to-End Object Detection with Transformers
    * Year: 26 May `2020`
    * Author: Nicolas Carion
    * Abstract: We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at [this https URL](https://github.com/facebookresearch/detr).
    * Comments:
        * > Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019, Wang et al., 2020). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. (ViT, 2020)
        * > DETR utilizes the Transformer decoder to model object detection as an end-to-end dictionary lookup problem with learnable queries, successfully removing the need for handcrafted processes such as NMS. (PVT, )
* [[Deformable DETR](https://arxiv.org/abs/2010.04159)] <!-- printed -->
    * Title: Deformable DETR: Deformable Transformers for End-to-End Object Detection
    * Year: 08 Oct `2020`
    * Authors: Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai
    * Abstract: DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach. Code is released at [this https URL](https://github.com/fundamentalvision/Deformable-DETR).
    * Comments:
        * > (2021, PVT) Based on DETR, deformable DETR [64] further introduces a deformable attention layer to focus on a sparse set of contextual elements which obtains fast convergence and better performance.

## Weekly-Supervised Learning

* [Is object localization for free? - Weakly-supervised learning with convolutional neural networks](https://ieeexplore.ieee.org/document/7298668)
    * Title: Is object localization for free? - Weakly-supervised learning with convolutional neural networks
