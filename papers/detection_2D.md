<span style="font-family:monospace">

# Papers in Computer Vision - 2D Object Detection

count: 55

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
* [SPPNet](https://arxiv.org/abs/1406.4729)
    * Title: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
    * Year: 18 Jun 2014
    * Author: Kaiming He
    * Abstract: Existing deep convolutional neural networks (CNNs) require a fixed-size (e.g., 224x224) input image. This requirement is "artificial" and may reduce the recognition accuracy for the images or sub-images of an arbitrary size/scale. In this work, we equip the networks with another pooling strategy, "spatial pyramid pooling", to eliminate the above requirement. The new network structure, called SPP-net, can generate a fixed-length representation regardless of image size/scale. Pyramid pooling is also robust to object deformations. With these advantages, SPP-net should in general improve all CNN-based image classification methods. On the ImageNet 2012 dataset, we demonstrate that SPP-net boosts the accuracy of a variety of CNN architectures despite their different designs. On the Pascal VOC 2007 and Caltech101 datasets, SPP-net achieves state-of-the-art classification results using a single full-image representation and no fine-tuning. The power of SPP-net is also significant in object detection. Using SPP-net, we compute the feature maps from the entire image only once, and then pool features in arbitrary regions (sub-images) to generate fixed-length representations for training the detectors. This method avoids repeatedly computing the convolutional features. In processing test images, our method is 24-102x faster than the R-CNN method, while achieving better or comparable accuracy on Pascal VOC 2007. In ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014, our methods rank #2 in object detection and #3 in image classification among all 38 teams. This manuscript also introduces the improvement made for this competition.
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

## Multi-Layers Detectors

* [ParseNet](https://arxiv.org/abs/1506.04579)
    * Title: ParseNet: Looking Wider to See Better
    * Year: 15 Jun `2015`
    * Author: Wei Liu
    * Abstract: We present a technique for adding global context to deep convolutional networks for semantic segmentation. The approach is simple, using the average feature for a layer to augment the features at each location. In addition, we study several idiosyncrasies of training, significantly increasing the performance of baseline networks (e.g. from FCN). When we add our proposed global feature, and a technique for learning normalization parameters, accuracy increases consistently even over our improved versions of the baselines. Our proposed approach, ParseNet, achieves state-of-the-art performance on SiftFlow and PASCAL-Context with small additional computational cost over baselines, and near current state-of-the-art performance on PASCAL VOC 2012 semantic segmentation with a simple approach. Code is available at [this https URL](https://github.com/weiliu89/caffe/tree/fcn).
* [Inside-Outside Net](https://arxiv.org/abs/1512.04143)
    * Title: Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
    * Year: 14 Dec `2015`
    * Author: Sean Bell
    * Abstract: It is well known that contextual and multi-scale representations are important for accurate visual recognition. In this paper we present the Inside-Outside Net (ION), an object detector that exploits information both inside and outside the region of interest. Contextual information outside the region of interest is integrated using spatial recurrent neural networks. Inside, we use skip pooling to extract information at multiple scales and levels of abstraction. Through extensive experiments we evaluate the design space and provide readers with an overview of what tricks of the trade are important. ION improves state-of-the-art on PASCAL VOC 2012 object detection from 73.9% to 76.4% mAP. On the new and more challenging MS COCO dataset, we improve state-of-art-the from 19.7% to 33.1% mAP. In the 2015 MS COCO Detection Challenge, our ION model won the Best Student Entry and finished 3rd place overall. As intuition suggests, our detection results provide strong evidence that context and multi-scale representations improve small object detection.
* [HyperNet](https://arxiv.org/abs/1604.00600)
    * Title: HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection
    * Year: 03 Apr `2016`
    * Author: Tao Kong
    * Abstract: Almost all of the current top-performing object detection networks employ region proposals to guide the search for object instances. State-of-the-art region proposal methods usually need several thousand proposals to get high recall, thus hurting the detection efficiency. Although the latest Region Proposal Network method gets promising detection accuracy with several hundred proposals, it still struggles in small-size object detection and precise localization (e.g., large IoU thresholds), mainly due to the coarseness of its feature maps. In this paper, we present a deep hierarchical network, namely HyperNet, for handling region proposal generation and object detection jointly. Our HyperNet is primarily based on an elaborately designed Hyper Feature which aggregates hierarchical feature maps first and then compresses them into a uniform space. The Hyper Features well incorporate deep but highly semantic, intermediate but really complementary, and shallow but naturally high-resolution features of the image, thus enabling us to construct HyperNet by sharing them both in generating proposals and detecting objects via an end-to-end joint training strategy. For the deep VGG16 model, our method achieves completely leading recall and state-of-the-art object detection accuracy on PASCAL VOC 2007 and 2012 using only 100 proposals per image. It runs with a speed of 5 fps (including all steps) on a GPU, thus having the potential for real-time processing.
* [Multi-Scale CNN](https://arxiv.org/abs/1607.07155)
    * Title: A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection
    * Year: 25 Jul `2016`
* [FPN](https://arxiv.org/abs/1612.03144)
    * Title: Feature Pyramid Networks for Object Detection
    * Year: 09 Dec `2016`
    * Author: Tsung-Yi Lin
    * Abstract: Feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive. In this paper, we exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A top-down architecture with lateral connections is developed for building high-level semantic feature maps at all scales. This architecture, called a Feature Pyramid Network (FPN), shows significant improvement as a generic feature extractor in several applications. Using FPN in a basic Faster R-CNN system, our method achieves state-of-the-art single-model results on the COCO detection benchmark without bells and whistles, surpassing all existing single-model entries including those from the COCO 2016 challenge winners. In addition, our method can run at 5 FPS on a GPU and thus is a practical and accurate solution to multi-scale object detection. Code will be made publicly available.

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
* [Fast R-CNN](https://arxiv.org/abs/1504.08083)
    * Title: Fast R-CNN
    * Year: 30 Apr `2015`
    * Author: Ross Girshick
    * Abstract: This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3x faster, tests 10x faster, and is more accurate. Fast R-CNN is implemented in Python and C++ (using Caffe) and is available under the open-source MIT License at [this https URL](https://github.com/rbgirshick/fast-rcnn).
* [Faster R-CNN](https://arxiv.org/abs/1506.01497)
    * Title: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
    * Year: 04 Jun `2015`
    * Author: Shaoqing Ren
    * Abstract: State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.
* [Mask R-CNN](https://arxiv.org/abs/1703.06870)
    * Title: Mask R-CNN
    * Year: 20 Mar 2017
    * Author: Kaiming He
    * Abstract: We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code has been made available at: [this https URL](https://github.com/facebookresearch/Detectron).
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

* [OHEM](https://arxiv.org/abs/1604.03540)
    * Title: Training Region-based Object Detectors with Online Hard Example Mining
    * Year: 12 Apr `2016`
    * Author: Abhinav Shrivastava
    * Abstract: The field of object detection has made significant advances riding on the wave of region-based ConvNets, but their training procedure still includes many heuristics and hyperparameters that are costly to tune. We present a simple yet surprisingly effective online hard example mining (OHEM) algorithm for training region-based ConvNet detectors. Our motivation is the same as it has always been -- detection datasets contain an overwhelming number of easy examples and a small number of hard examples. Automatic selection of these hard examples can make training more effective and efficient. OHEM is a simple and intuitive algorithm that eliminates several heuristics and hyperparameters in common use. But more importantly, it yields consistent and significant boosts in detection performance on benchmarks like PASCAL VOC 2007 and 2012. Its effectiveness increases as datasets become larger and more difficult, as demonstrated by the results on the MS COCO dataset. Moreover, combined with complementary advances in the field, OHEM leads to state-of-the-art results of 78.9% and 76.3% mAP on PASCAL VOC 2007 and 2012 respectively.
* (20 May 2016) [R-FCN](https://arxiv.org/abs/1605.06409) (R-FCN: Object Detection via Region-based Fully Convolutional Networks)
* (29 Aug 2016) [PVANET](https://arxiv.org/abs/1608.08021) (PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection)
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