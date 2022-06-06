<span style="font-family:monospace">

# Papers in Computer Vision - Detection

count: 52

## General

* [OverFeat](https://arxiv.org/abs/1312.6229)
    * Title: OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
    * Year: 21 Dec `2013`
    * Author: Pierre Sermanet
    * Abstract: We present an integrated framework for using Convolutional Networks for classification, localization and detection. We show how a multiscale and sliding window approach can be efficiently implemented within a ConvNet. We also introduce a novel deep learning approach to localization by learning to predict object boundaries. Bounding boxes are then accumulated rather than suppressed in order to increase detection confidence. We show that different tasks can be learned simultaneously using a single shared network. This integrated framework is the winner of the localization task of the ImageNet Large Scale Visual Recognition Challenge 2013 (ILSVRC2013) and obtained very competitive results for the detection and classifications tasks. In post-competition work, we establish a new state of the art for the detection task. Finally, we release a feature extractor from our best model called OverFeat.
* [MultiBox](https://arxiv.org/abs/1312.2249)
    * Title: Scalable Object Detection using Deep Neural Networks
    * Year: 08 Dec `2013`
    * Author: Dumitru Erhan
    * Abstract: Deep convolutional neural networks have recently achieved state-of-the-art performance on a number of image recognition benchmarks, including the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC-2012). The winning model on the localization sub-task was a network that predicts a single bounding box and a confidence score for each object category in the image. Such a model captures the whole-image context around the objects but cannot handle multiple instances of the same object in the image without naively replicating the number of outputs for each instance. In this work, we propose a saliency-inspired neural network model for detection, which predicts a set of class-agnostic bounding boxes along with a single score for each box, corresponding to its likelihood of containing any object of interest. The model naturally handles a variable number of instances for each class and allows for cross-class generalization at the highest levels of the network. We are able to obtain competitive recognition performance on VOC2007 and ILSVRC2012, while using only the top few predicted locations in each image and a small number of neural network evaluations.
* (18 Jun 2014) [SPPNet](https://arxiv.org/abs/1406.4729) (Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition)
* [DPM](https://arxiv.org/abs/1409.5403)
    * Title: Deformable Part Models are Convolutional Neural Networks
    * Year: 18 Sep `2014`
    * Author: Ross Girshick
    * Abstract: Deformable part models (DPMs) and convolutional neural networks (CNNs) are two widely used tools for visual recognition. They are typically viewed as distinct approaches: DPMs are graphical models (Markov random fields), while CNNs are "black-box" non-linear classifiers. In this paper, we show that a DPM can be formulated as a CNN, thus providing a novel synthesis of the two ideas. Our construction involves unrolling the DPM inference algorithm and mapping each step to an equivalent (and at times novel) CNN layer. From this perspective, it becomes natural to replace the standard image features used in DPM with a learned feature extractor. We call the resulting model DeepPyramid DPM and experimentally validate it on PASCAL VOC. DeepPyramid DPM significantly outperforms DPMs based on histograms of oriented gradients features (HOG) and slightly outperforms a comparable version of the recently introduced R-CNN detection system, while running an order of magnitude faster.
* [DPM](https://ieeexplore.ieee.org/document/5255236)
    * Title: Object Detection with Discriminatively Trained Part-Based Models
    * Year: `2010`
    * Author: Pedro F. Felzenszwalb
    * Abstract: We describe an object detection system based on mixtures of multiscale deformable part models. Our system is able to represent highly variable object classes and achieves state-of-the-art results in the PASCAL object detection challenges. While deformable part models have become quite popular, their value had not been demonstrated on difficult benchmarks such as the PASCAL data sets. Our system relies on new methods for discriminative training with partially labeled data. We combine a margin-sensitive approach for data-mining hard negative examples with a formalism we call latent SVM. A latent SVM is a reformulation of MI--SVM in terms of latent variables. A latent SVM is semiconvex, and the training problem becomes convex once latent information is specified for the positive examples. This leads to an iterative training algorithm that alternates between fixing latent values for positive examples and optimizing the latent SVM objective function.
* (07 May 2015) [Multi-Region CNN](https://arxiv.org/abs/1505.01749) (Object detection via a multi-region & semantic segmentation-aware CNN model)
* (08 May 2015) [DeepBox](https://arxiv.org/abs/1505.02146) (DeepBox: Learning Objectness with Convolutional Networks)
* (25 Jun 2015) [AttentionNet](https://arxiv.org/abs/1506.07704) (AttentionNet: Aggregating Weak Directions for Accurate Object Detection)
* (15 Oct 2015) [DeepProposal](https://arxiv.org/abs/1510.04445) (DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers)
* (03 Dec 2014) [Scalable, High-Quality Object Detection](https://arxiv.org/abs/1412.1441)
* (07 Jan 2019) [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)
* (16 Apr 2019) [CenterNet](https://arxiv.org/abs/1904.07850) (Objects as Points)
* (03 Jun 2020) [DetectoRS](https://arxiv.org/abs/2006.02334): Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolutions
* (23 Mar 2021) [Robust and Accurate Object Detection via Adversarial Learning](https://arxiv.org/abs/2103.13886)

## SSD and its Variants

* [SSD](https://arxiv.org/abs/1512.02325)
    * Title: SSD: Single Shot MultiBox Detector
    * Year: 08 Dec `2015`
    * Author: Wei Liu
    * Abstract: We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For 300x300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for 500x500 input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model. Code is available at [this https URL](https://github.com/weiliu89/caffe/tree/ssd).
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

----------------------------------------------------------------------------------------------------
## R-CNN Series

* [R-CNN](https://arxiv.org/abs/1311.2524)
    * Title: Rich feature hierarchies for accurate object detection and semantic segmentation
    * Year: 11 Nov `2013`
    * Author: Ross Girshick
    * Abstract: Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012---achieving a mAP of 53.3%. Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also compare R-CNN to OverFeat, a recently proposed sliding-window detector based on a similar CNN architecture. We find that R-CNN outperforms OverFeat by a large margin on the 200-class ILSVRC2013 detection dataset. Source code for the complete system is available at [this http URL](http://www.cs.berkeley.edu/~rbg/rcnn).
* (30 Apr 2015) [Fast R-CNN](https://arxiv.org/abs/1504.08083) (Fast R-CNN)
* (04 Jun 2015) [Faster R-CNN](https://arxiv.org/abs/1506.01497) (Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks)
* (20 Mar 2017) [Mask R-CNN](https://arxiv.org/abs/1703.06870) (Mask R-CNN)
* [Cascade R-CNN](https://arxiv.org/abs/1712.00726)
    * Title: Cascade R-CNN: Delving into High Quality Object Detection
    * Year: 03 Dec `2017`
    * Author: Zhaowei Cai
    * Abstract: In object detection, an intersection over union (IoU) threshold is required to define positives and negatives. An object detector, trained with low IoU threshold, e.g. 0.5, usually produces noisy detections. However, detection performance tends to degrade with increasing the IoU thresholds. Two main factors are responsible for this: 1) overfitting during training, due to exponentially vanishing positive samples, and 2) inference-time mismatch between the IoUs for which the detector is optimal and those of the input hypotheses. A multi-stage object detection architecture, the Cascade R-CNN, is proposed to address these problems. It consists of a sequence of detectors trained with increasing IoU thresholds, to be sequentially more selective against close false positives. The detectors are trained stage by stage, leveraging the observation that the output of a detector is a good distribution for training the next higher quality detector. The resampling of progressively improved hypotheses guarantees that all detectors have a positive set of examples of equivalent size, reducing the overfitting problem. The same cascade procedure is applied at inference, enabling a closer match between the hypotheses and the detector quality of each stage. A simple implementation of the Cascade R-CNN is shown to surpass all single-model object detectors on the challenging COCO dataset. Experiments also show that the Cascade R-CNN is widely applicable across detector architectures, achieving consistent gains independently of the baseline detector strength. The code will be made available at [this https URL](https://github.com/zhaoweicai/cascade-rcnn).
* (13 Apr 2020) [Dynamic R-CNN](https://arxiv.org/abs/2004.06002): Towards High Quality Object Detection via Dynamic Training

----------------------------------------------------------------------------------------------------
## YOLO Series

* (08 Jun 2015) [YOLOv1](https://arxiv.org/abs/1506.02640) (You Only Look Once: Unified, Real-Time Object Detection)
* (25 Dec 2016) [YOLOv2](https://arxiv.org/abs/1612.08242) (YOLO9000: Better, Faster, Stronger)
* (08 Apr 2018) [YOLOv3](https://arxiv.org/abs/1804.02767) (An Incremental Improvement)
* (23 Apr 2020) [YOLOv4](https://arxiv.org/abs/2004.10934) (Optimal Speed and Accuracy of Object Detections)
* (16 Nov 2020) [Scaled-YOLOv4](https://arxiv.org/abs/2011.08036) (Scaled-YOLOv4: Scaling Cross Stage Partial Network)
* (17 Mar 2021) [YOLOF](https://arxiv.org/abs/2103.09460) (You Only Look One-level Feature)
* (10 May 2021) [YOLOR](https://arxiv.org/abs/2105.04206) (You Only Learn One Representation)
* (01 Jun 2021) [YOLOS](https://arxiv.org/abs/2106.00666) (You Only Look at One Sequence)
* (18 Jul 2021) [YOLOX](https://arxiv.org/abs/2107.08430) (Exceeding YOLO Series in 2021)

----------------------------------------------------------------------------------------------------
## Improvements

* [FCN](https://arxiv.org/abs/1411.4038)
    * Title: Fully Convolutional Networks for Semantic Segmentation
    * Year: 14 Nov `2014`
    * Author: Jonathan Long
    * Abstract: Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build "fully convolutional" networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet, the VGG net, and GoogLeNet) into fully convolutional networks and transfer their learned representations by fine-tuning to the segmentation task. We then define a novel architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional network achieves state-of-the-art segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes one third of a second for a typical image.
* [OHEM](https://arxiv.org/abs/1604.03540)
    * Title: Training Region-based Object Detectors with Online Hard Example Mining
    * Year: 12 Apr `2016`
    * Author: Abhinav Shrivastava
    * Abstract: The field of object detection has made significant advances riding on the wave of region-based ConvNets, but their training procedure still includes many heuristics and hyperparameters that are costly to tune. We present a simple yet surprisingly effective online hard example mining (OHEM) algorithm for training region-based ConvNet detectors. Our motivation is the same as it has always been -- detection datasets contain an overwhelming number of easy examples and a small number of hard examples. Automatic selection of these hard examples can make training more effective and efficient. OHEM is a simple and intuitive algorithm that eliminates several heuristics and hyperparameters in common use. But more importantly, it yields consistent and significant boosts in detection performance on benchmarks like PASCAL VOC 2007 and 2012. Its effectiveness increases as datasets become larger and more difficult, as demonstrated by the results on the MS COCO dataset. Moreover, combined with complementary advances in the field, OHEM leads to state-of-the-art results of 78.9% and 76.3% mAP on PASCAL VOC 2007 and 2012 respectively.
* (20 May 2016) [R-FCN](https://arxiv.org/abs/1605.06409) (R-FCN: Object Detection via Region-based Fully Convolutional Networks)
* (25 Jul 2016) [Multi-Scale CNN](https://arxiv.org/abs/1607.07155) (A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection)
* (29 Aug 2016) [PVANET](https://arxiv.org/abs/1608.08021) (PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection)
* (09 Dec 2016) [FPN](https://arxiv.org/abs/1612.03144) (Feature Pyramid Networks for Object Detection)
* (17 Mar 2017) [Deformable](https://arxiv.org/abs/1703.06211) (Deformable Convolutional Networks)
* (30 Mar 2017) [DeNet](https://arxiv.org/abs/1703.10295) (DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling)
* (06 Jul 2017) [RON](https://arxiv.org/abs/1707.01691) (RON: Reverse Connection with Objectness Prior Networks for Object Detection)
* (09 Aug 2017) [CoupleNet](https://arxiv.org/abs/1708.02863) (CoupleNet: Coupling Global Structure with Local Parts for Object Detection)
* (07 Aug 2017) [RetinaNet](https://arxiv.org/abs/1708.02002) (Focal Loss for Dense Object Detection)
* (03 Aug 2017) [DSOD](https://arxiv.org/abs/1708.01241) (DSOD: Learning Deeply Supervised Object Detectors from Scratch)
* (20 Nov 2019) [EfficientDet](https://arxiv.org/abs/1911.09070) (EfficientDet: Scalable and Efficient Object Detection)

----------------------------------------------------------------------------------------------------
## Anchor-Free Frameworks

* (03 Aug 2018) [CornerNet](https://arxiv.org/abs/1808.01244) (CornerNet: Detecting Objects as Paired Keypoints)
* (23 Jan 2019) [ExtremeNet](https://arxiv.org/abs/1901.08043) (Bottom-up Object Detection by Grouping Extreme and Center Points)
* (02 Apr 2019) [FCOS](https://arxiv.org/abs/1904.01355) (FCOS: Fully Convolutional One-Stage Object Detection)
* (17 Apr 2019) [CenterNet](https://arxiv.org/abs/1904.08189) (CenterNet: Keypoint Triplets for Object Detection)
* (02 Mar 2019) [FSAF](https://arxiv.org/abs/1903.00621) (Feature Selective Anchor-Free Module for Single-Shot Object Detection)
* (16 Apr 2019) [NAS-FPN](https://arxiv.org/abs/1904.07392) (NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection)
* (26 Mar 2019) [DetNAS](https://arxiv.org/abs/1903.10979) (DetNAS: Backbone Search for Object Detection)
* (28 May 2019) [EfficientNet](https://arxiv.org/abs/1905.11946) (EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)
* (26 May 2020) [DETR](https://arxiv.org/abs/2005.12872) (End-to-End Object Detection with Transformers)
