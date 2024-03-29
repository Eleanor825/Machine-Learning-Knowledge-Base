# [Papers][Vision] 2D Segmentation <!-- omit in toc -->

count=113

## Table of Contents <!-- omit in toc -->

- [1. Unknown](#1-unknown)
- [2. classify region proposals](#2-classify-region-proposals)
- [3. Superpixels (Learning Hierarchical Features for Scene Labeling, 2013)](#3-superpixels-learning-hierarchical-features-for-scene-labeling-2013)
- [4. Superpixels (LRR, 2016) (2)](#4-superpixels-lrr-2016-2)
- [5. Semantic Segmentation](#5-semantic-segmentation)
  - [5.1. Unclassified](#51-unclassified)
  - [5.2. (2015, FCN) Patchwise Training (count=5)](#52-2015-fcn-patchwise-training-count5)
  - [5.3. Fully Convolutional Networks (FCN)](#53-fully-convolutional-networks-fcn)
  - [5.4. FCN based models (Attention to Scale, 2015) (6)](#54-fcn-based-models-attention-to-scale-2015-6)
  - [5.5. The U-Net Family](#55-the-u-net-family)
  - [5.6. Encoder-Decoder Architectures](#56-encoder-decoder-architectures)
  - [5.7. Increase feature resolution (Panoptic FPN, 2019) (4)](#57-increase-feature-resolution-panoptic-fpn-2019-4)
  - [5.8. Multi Scale Architectures (Attention to Scale, 2015) (1 + 4 + 1 + 4 + 1)](#58-multi-scale-architectures-attention-to-scale-2015-1--4--1--4--1)
  - [5.9. Multi-Scale Architectures (2016, PSPNet) (5)](#59-multi-scale-architectures-2016-pspnet-5)
  - [5.10. Efficiency](#510-efficiency)
- [6. Graphical-Model-Based Approaches](#6-graphical-model-based-approaches)
  - [6.1. Unknown](#61-unknown)
  - [6.2. (2014, DeepLabV1) Conditional Random Fields (CRF) (count=3)](#62-2014-deeplabv1-conditional-random-fields-crf-count3)
  - [6.3. MAP and CRF (Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, 2012) (count=5+5)](#63-map-and-crf-efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials-2012-count55)
  - [6.4. (2015, SegNet) MAP and CRF (count=2)](#64-2015-segnet-map-and-crf-count2)
  - [6.5. Fully Connected CRFs (Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, 2012) (4)](#65-fully-connected-crfs-efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials-2012-4)
  - [6.6. DeepLab Family](#66-deeplab-family)
- [7. Attention Mechanism](#7-attention-mechanism)
  - [7.1. (2022, SegNeXt) (count=6)](#71-2022-segnext-count6)
- [8. Transformer-Architectures-Based Approaches](#8-transformer-architectures-based-approaches)
  - [8.1. (2022, SegNeXt) (count=8)](#81-2022-segnext-count8)
  - [8.2. Others](#82-others)
- [9. Instance Segmentation](#9-instance-segmentation)
  - [9.1. Segment Proposals](#91-segment-proposals)
  - [9.2. Fully Convolutional Instance Segmentation (FCIS) Systems](#92-fully-convolutional-instance-segmentation-fcis-systems)
- [10. Multitask Learning (Panoptic Segmentation, 2018) (3)](#10-multitask-learning-panoptic-segmentation-2018-3)
- [11. Panoptic Segmentation](#11-panoptic-segmentation)
- [12. Scene Parsing](#12-scene-parsing)
- [13. weakly supervised (2015, DeconvNet)](#13-weakly-supervised-2015-deconvnet)

----------------------------------------------------------------------------------------------------

## 1. Unknown

* [Multi-Digit Recognition Using a Space Displacement Neural Network](https://papers.nips.cc/paper/1991/hash/6e2713a6efee97bacb63e52c54f0ada0-Abstract.html)
    * Title: Multi-Digit Recognition Using a Space Displacement Neural Network
    * Year: `1991`
    * Authors: Ofer Matan, Christopher J. C. Burges, Yann LeCun, John Denker
    * Abstract: We present a feed-forward network architecture for recognizing an uncon(cid:173) strained handwritten multi-digit string. This is an extension of previous work on recognizing isolated digits. In this architecture a single digit rec(cid:173) ognizer is replicated over the input. The output layer of the network is coupled to a Viterbi alignment module that chooses the best interpretation of the input. Training errors are propagated through the Viterbi module. The novelty in this procedure is that segmentation is done on the feature maps developed in the Space Displacement Neural Network (SDNN) rather than the input (pixel) space.
* [[Image Segmentation with Cascaded Hierarchical Models and Logistic Disjunctive Normal Networks](https://ieeexplore.ieee.org/document/6751380)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6751380)]
    * Title: Image Segmentation with Cascaded Hierarchical Models and Logistic Disjunctive Normal Networks
* [[Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture](https://arxiv.org/abs/1411.4734)]
    [[pdf](https://arxiv.org/pdf/1411.4734.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1411.4734/)]
    * Title: Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture
    * Year: 18 Nov `2014`
    * Authors: David Eigen, Rob Fergus
    * Abstract: In this paper we address three different computer vision tasks using a single basic architecture: depth prediction, surface normal estimation, and semantic labeling. We use a multiscale convolutional network that is able to adapt easily to each task using only small modifications, regressing from the input image to the output map directly. Our method progressively refines predictions using a sequence of scales, and captures many image details without any superpixels or low-level segmentation. We achieve state-of-the-art performance on benchmarks for all three tasks.
    * Comments:
        * > More recently, the segmentation-free techniques of (Long et al., 2014; Eigen & Fergus, 2014) directly apply DCNNs to the whole image in a sliding window fashion, replacing the last fully connected layers of a DCNN by convolutional layers. In order to deal with the spatial localization issues outlined in the beginning of the introduction, Long et al. (2014) upsample and concatenate the scores from inter-mediate feature maps, while Eigen & Fergus (2014) refine the prediction result from coarse to fine by propagating the coarse results to another DCNN. (DeepLabv1, 2014)
        * Introduced median frequency balancing.
* [[Semantic texton forests for image categorization and segmentation](https://ieeexplore.ieee.org/document/4587503)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4587503)]
    * Title: Semantic texton forests for image categorization and segmentation
    * Year: Jamie Shotton; Matthew Johnson; Roberto Cipolla


## 2. classify region proposals

> (2015, DeconvNet) Some algorithms [3, 9, 10] classify region proposals and refine the labels in the image-level segmentation map to obtain the final segmentation.

* [[Convolutional Feature Masking for Joint Object and Stuff Segmentation](https://arxiv.org/abs/1412.1283)]
    [[pdf](https://arxiv.org/pdf/1412.1283.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.1283/)]
    * Title: Convolutional Feature Masking for Joint Object and Stuff Segmentation
    * Year: 03 Dec `2014`
    * Authors: Jifeng Dai, Kaiming He, Jian Sun
    * Abstract: The topic of semantic segmentation has witnessed considerable progress due to the powerful features learned by convolutional neural networks (CNNs). The current leading approaches for semantic segmentation exploit shape information by extracting CNN features from masked image regions. This strategy introduces artificial boundaries on the images and may impact the quality of the extracted features. Besides, the operations on the raw image domain require to compute thousands of networks on a single image, which is time-consuming. In this paper, we propose to exploit shape information via masking convolutional features. The proposal segments (e.g., super-pixels) are treated as masks on the convolutional feature maps. The CNN features of segments are directly masked out from these maps and used to train classifiers for recognition. We further propose a joint method to handle objects and "stuff" (e.g., grass, sky, water) in the same framework. State-of-the-art results are demonstrated on benchmarks of PASCAL VOC and new PASCAL-CONTEXT, with a compelling computational speed.
* Simultaneous Detection and Segmentation
* Hypercolumns for Object Segmentation and Fine-grained Localization

## 3. Superpixels (Learning Hierarchical Features for Scene Labeling, 2013)

* [[Efficient Graph-Based Image Segmentation](https://link.springer.com/article/10.1023/B:VISI.0000022288.19776.77)]
    [[pdf](https://link.springer.com/content/pdf/10.1023/B:VISI.0000022288.19776.77.pdf)]
    * Title: Efficient Graph-Based Image Segmentation
    * Year: `2004`
    * Authors: Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    * Abstract: This paper addresses the problem of segmenting an image into regions. We define a predicate for measuring the evidence for a boundary between two regions using a graph-based representation of the image. We then develop an efficient segmentation algorithm based on this predicate, and show that although this algorithm makes greedy decisions it produces segmentations that satisfy global properties. We apply the algorithm to image segmentation using two different kinds of local neighborhoods in constructing the graph, and illustrate the results with both real and synthetic images. The algorithm runs in time nearly linear in the number of graph edges and is also fast in practice. An important characteristic of the method is its ability to preserve detail in low-variability image regions while ignoring detail in high-variability regions.
* [[Class segmentation and object localization with superpixel neighborhoods](https://ieeexplore.ieee.org/document/5459175)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5459175)]
    * Title: Class segmentation and object localization with superpixel neighborhoods
    * Year: 29 July `2010`
    * Authors: Brian Fulkerson; Andrea Vedaldi; Stefano Soatto
    * Abstract: We propose a method to identify and localize object classes in images. Instead of operating at the pixel level, we advocate the use of superpixels as the basic unit of a class segmentation or pixel localization scheme. To this end, we construct a classifier on the histogram of local features found in each superpixel. We regularize this classifier by aggregating histograms in the neighborhood of each superpixel and then refine our results further by using the classifier in a conditional random field operating on the superpixel graph. Our proposed method exceeds the previously published state-of-the-art on two challenging datasets: Graz-02 and the PASCAL VOC 2007 Segmentation Challenge.
* [[Multi-Class Segmentation with Relative Location Prior](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.145.6337)]
    [[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.6337&rep=rep1&type=pdf)]
    * Title: Multi-Class Segmentation with Relative Location Prior
    * Year: `2008`
    * Authors: Stephen Gould , Jim Rodgers , David Cohen , Gal Elidan , Daphne Koller
    * Abstract: Multi-class image segmentation has made significant advances in recent years through the combination of local and global features. One important type of global feature is that of inter-class spatial relationships. For example, identifying "tree" pixels indicates that pixels above and to the sides are more likely to be "sky" whereas pixels below are more likely to be "grass." Incorporating such global information across the entire image and between all classes is a computational challenge as it is image-dependent, and hence, cannot be precomputed. In this work we propose a method for capturing global information from inter-class spatial relationships and encoding it as a local feature. We employ a two-stage classification process to label all image pixels. First, we generate predictions which are used to compute a local relative location feature from learned relative location maps. In the second stage, we combine this with appearance-based features to provide a final segmentation. We compare our results to recent published results on several multiclass image segmentation databases and show that the incorporation of relative location information allows us to significantly outperform the current state-of-the-art.

## 4. Superpixels (LRR, 2016) (2)

* Convolutional Feature Masking for Joint Object and Stuff Segmentation
* [[Feedforward semantic segmentation with zoom-out features](https://arxiv.org/abs/1412.0774)]
    [[pdf](https://arxiv.org/pdf/1412.0774.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.0774/)]
    * Title: Feedforward semantic segmentation with zoom-out features
    * Year: 02 Dec `2014`
    * Authors: Mohammadreza Mostajabi, Payman Yadollahpour, Gregory Shakhnarovich
    * Abstract: We introduce a purely feed-forward architecture for semantic segmentation. We map small image elements (superpixels) to rich feature representations extracted from a sequence of nested regions of increasing extent. These regions are obtained by "zooming out" from the superpixel all the way to scene-level resolution. This approach exploits statistical structure in the image and in the label space without setting up explicit structured prediction mechanisms, and thus avoids complex and expensive inference. Instead superpixels are classified by a feedforward multilayer network. Our architecture achieves new state of the art performance in semantic segmentation, obtaining 64.4% average accuracy on the PASCAL VOC 2012 test set.
    * Comments:
        * > Mostajabi et al. [41] classified a superpixel with features extracted at zoom-out spatial levels from a small proximal neighborhood to the whole image region. (Attention to Scale, 2015)

## 5. Semantic Segmentation

### 5.1. Unclassified

* [[GFF](https://arxiv.org/abs/1904.01803)]
    [[pdf](https://arxiv.org/pdf/1904.01803.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.01803/)]
    * Title: GFF: Gated Fully Fusion for Semantic Segmentation
    * Year: 03 Apr `2019`
    * Authors: Xiangtai Li, Houlong Zhao, Lei Han, Yunhai Tong, Kuiyuan Yang
    * Abstract: Semantic segmentation generates comprehensive understanding of scenes through densely predicting the category for each pixel. High-level features from Deep Convolutional Neural Networks already demonstrate their effectiveness in semantic segmentation tasks, however the coarse resolution of high-level features often leads to inferior results for small/thin objects where detailed information is important. It is natural to consider importing low level features to compensate for the lost detailed information in high-level features.Unfortunately, simply combining multi-level features suffers from the semantic gap among them. In this paper, we propose a new architecture, named Gated Fully Fusion (GFF), to selectively fuse features from multiple levels using gates in a fully connected way. Specifically, features at each level are enhanced by higher-level features with stronger semantics and lower-level features with more details, and gates are used to control the propagation of useful information which significantly reduces the noises during fusion. We achieve the state of the art results on four challenging scene parsing datasets including Cityscapes, Pascal Context, COCO-stuff and ADE20K.
* [[Learning Rich Features from RGB-D Images for Object Detection and Segmentation](https://arxiv.org/abs/1407.5736)]
    [[pdf](https://arxiv.org/pdf/1407.5736.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1407.5736/)]
    * Title: Learning Rich Features from RGB-D Images for Object Detection and Segmentation
    * Year: 22 Jul `2014`
    * Authors: Saurabh Gupta, Ross Girshick, Pablo Arbel�ez, Jitendra Malik
    * Abstract: In this paper we study the problem of object detection for RGB-D images using semantically rich image and depth features. We propose a new geocentric embedding for depth images that encodes height above ground and angle with gravity for each pixel in addition to the horizontal disparity. We demonstrate that this geocentric embedding works better than using raw depth images for learning feature representations with convolutional neural networks. Our final object detection system achieves an average precision of 37.3%, which is a 56% relative improvement over existing methods. We then focus on the task of instance segmentation where we label pixels belonging to object instances found by our detector. For this task, we propose a decision forest approach that classifies pixels in the detection window as foreground or background using a family of unary and binary tests that query shape and geocentric pose features. Finally, we use the output from our object detectors in an existing superpixel classification framework for semantic scene segmentation and achieve a 24% relative improvement over current state-of-the-art for the object categories that we study. We believe advances such as those represented in this paper will facilitate the use of perception in fields like robotics.
* [[RefineNet](https://arxiv.org/abs/1611.06612)]
    [[pdf](https://arxiv.org/pdf/1611.06612.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1611.06612/)]
    * Title: RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
    * Year: 20 Nov `2016`
    * Authors: Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid
    * Institutions: [The University of Adelaide], [Australian Centre for Robotic Vision]
    * Abstract: Recently, very deep convolutional neural networks (CNNs) have shown outstanding performance in object recognition and have also been the first choice for dense classification problems such as semantic segmentation. However, repeated subsampling operations like pooling or convolution striding in deep CNNs lead to a significant decrease in the initial image resolution. Here, we present RefineNet, a generic multi-path refinement network that explicitly exploits all the information available along the down-sampling process to enable high-resolution prediction using long-range residual connections. In this way, the deeper layers that capture high-level semantic features can be directly refined using fine-grained features from earlier convolutions. The individual components of RefineNet employ residual connections following the identity mapping mindset, which allows for effective end-to-end training. Further, we introduce chained residual pooling, which captures rich background context in an efficient manner. We carry out comprehensive experiments and set new state-of-the-art results on seven public datasets. In particular, we achieve an intersection-over-union score of 83.4 on the challenging PASCAL VOC 2012 dataset, which is the best reported result to date.
    * Comments:
        * > Lin et al. [9] (RefineNet) propose a multi-path refinement network that exploits all the information available along the downsampling process to enable high-resolution predictions using long-range residual connections. (2017, ERFNet)
* [[High-performance Semantic Segmentation Using Very Deep Fully Convolutional Networks](https://arxiv.org/abs/1604.04339)]
    [[pdf](https://arxiv.org/pdf/1604.04339.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1604.04339/)]
    * Title: High-performance Semantic Segmentation Using Very Deep Fully Convolutional Networks
    * Year: 15 Apr `2016`
    * Authors: Zifeng Wu, Chunhua Shen, Anton van den Hengel
    * Abstract: We propose a method for high-performance semantic image segmentation (or semantic pixel labelling) based on very deep residual networks, which achieves the state-of-the-art performance. A few design factors are carefully considered to this end. We make the following contributions. (i) First, we evaluate different variations of a fully convolutional residual network so as to find the best configuration, including the number of layers, the resolution of feature maps, and the size of field-of-view. Our experiments show that further enlarging the field-of-view and increasing the resolution of feature maps are typically beneficial, which however inevitably leads to a higher demand for GPU memories. To walk around the limitation, we propose a new method to simulate a high resolution network with a low resolution network, which can be applied during training and/or testing. (ii) Second, we propose an online bootstrapping method for training. We demonstrate that online bootstrapping is critically important for achieving good accuracy. (iii) Third we apply the traditional dropout to some of the residual blocks, which further improves the performance. (iv) Finally, our method achieves the currently best mean intersection-over-union 78.3\% on the PASCAL VOC 2012 dataset, as well as on the recent dataset Cityscapes.
* [[BiSeNetV1](https://arxiv.org/abs/1808.00897)]
    [[pdf](https://arxiv.org/pdf/1808.00897.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1808.00897/)]
    * Title: BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
    * Year: 02 Aug `2018`
    * Authors: Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, Nong Sang
    * Institutions: [National Key Laboratory of Science and Technology on Multispectral Information Processing, School of Automation, Huazhong University of Science & Technology,China], [Key Laboratory of Machine Perception, Peking University, China], [Megvii Inc. (Face++), China]
    * Abstract: Semantic segmentation requires both rich spatial information and sizeable receptive field. However, modern approaches usually compromise spatial resolution to achieve real-time inference speed, which leads to poor performance. In this paper, we address this dilemma with a novel Bilateral Segmentation Network (BiSeNet). We first design a Spatial Path with a small stride to preserve the spatial information and generate high-resolution features. Meanwhile, a Context Path with a fast downsampling strategy is employed to obtain sufficient receptive field. On top of the two paths, we introduce a new Feature Fusion Module to combine features efficiently. The proposed architecture makes a right balance between the speed and segmentation performance on Cityscapes, CamVid, and COCO-Stuff datasets. Specifically, for a 2048x1024 input, we achieve 68.4% Mean IOU on the Cityscapes test dataset with speed of 105 FPS on one NVIDIA Titan XP card, which is significantly faster than the existing methods with comparable performance.
* [[BiSeNetV2](https://arxiv.org/abs/2004.02147)]
    [[pdf](https://arxiv.org/pdf/2004.02147.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2004.02147/)]
    * Title: BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation
    * Year: 05 Apr `2020`
    * Authors: Changqian Yu, Changxin Gao, Jingbo Wang, Gang Yu, Chunhua Shen, Nong Sang
    * Institutions: [National Key Laboratory of Science and Technology on Multispectral Information Processing, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan, China], [The University of Adelaide, Australia], [The Chinese University of Hong Kong], [Tencent]
    * Abstract: The low-level details and high-level semantics are both essential to the semantic segmentation task. However, to speed up the model inference, current approaches almost always sacrifice the low-level details, which leads to a considerable accuracy decrease. We propose to treat these spatial details and categorical semantics separately to achieve high accuracy and high efficiency for realtime semantic segmentation. To this end, we propose an efficient and effective architecture with a good trade-off between speed and accuracy, termed Bilateral Segmentation Network (BiSeNet V2). This architecture involves: (i) a Detail Branch, with wide channels and shallow layers to capture low-level details and generate high-resolution feature representation; (ii) a Semantic Branch, with narrow channels and deep layers to obtain high-level semantic context. The Semantic Branch is lightweight due to reducing the channel capacity and a fast-downsampling strategy. Furthermore, we design a Guided Aggregation Layer to enhance mutual connections and fuse both types of feature representation. Besides, a booster training strategy is designed to improve the segmentation performance without any extra inference cost. Extensive quantitative and qualitative evaluations demonstrate that the proposed architecture performs favourably against a few state-of-the-art real-time semantic segmentation approaches. Specifically, for a 2,048x1,024 input, we achieve 72.6% Mean IoU on the Cityscapes test set with a speed of 156 FPS on one NVIDIA GeForce GTX 1080 Ti card, which is significantly faster than existing methods, yet we achieve better segmentation accuracy.
* [[LinkNet](https://arxiv.org/abs/1707.03718)]
    [[pdf](https://arxiv.org/pdf/1707.03718.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1707.03718/)]
    * Title: LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
    * Year: 14 Jun `2017`
    * Authors: Abhishek Chaurasia, Eugenio Culurciello
    * Abstract: Pixel-wise semantic segmentation for visual scene understanding not only needs to be accurate, but also efficient in order to find any use in real-time application. Existing algorithms even though are accurate but they do not focus on utilizing the parameters of neural network efficiently. As a result they are huge in terms of parameters and number of operations; hence slow too. In this paper, we propose a novel deep neural network architecture which allows it to learn without any significant increase in number of parameters. Our network uses only 11.5 million parameters and 21.2 GFLOPs for processing an image of resolution 3x640x360. It gives state-of-the-art performance on CamVid and comparable results on Cityscapes dataset. We also compare our networks processing time on NVIDIA GPU and embedded system device with existing state-of-the-art architectures for different image resolutions.
* [[FlowFormer: A Transformer Architecture for Optical Flow](https://arxiv.org/abs/2203.16194)]
    [[pdf](https://arxiv.org/pdf/2203.16194.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2203.16194/)]
    * Title: FlowFormer: A Transformer Architecture for Optical Flow
    * Year: 30 Mar `2022`
    * Authors: Zhaoyang Huang, Xiaoyu Shi, Chao Zhang, Qiang Wang, Ka Chun Cheung, Hongwei Qin, Jifeng Dai, Hongsheng Li
    * Abstract: We introduce optical Flow transFormer, dubbed as FlowFormer, a transformer-based neural network architecture for learning optical flow. FlowFormer tokenizes the 4D cost volume built from an image pair, encodes the cost tokens into a cost memory with alternate-group transformer (AGT) layers in a novel latent space, and decodes the cost memory via a recurrent transformer decoder with dynamic positional cost queries. On the Sintel benchmark, FlowFormer achieves 1.159 and 2.088 average end-point-error (AEPE) on the clean and final pass, a 16.5% and 15.5% error reduction from the best published result (1.388 and 2.47). Besides, FlowFormer also achieves strong generalization performance. Without being trained on Sintel, FlowFormer achieves 1.01 AEPE on the clean pass of Sintel training set, outperforming the best published result (1.29) by 21.7%.
* [[Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247)]
    [[pdf](https://arxiv.org/pdf/1406.6247.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1406.6247/)]
    * Title: Recurrent Models of Visual Attention
    * Year: 24 Jun `2014`
    * Authors: Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu
    * Abstract: Applying convolutional neural networks to large images is computationally expensive because the amount of computation scales linearly with the number of image pixels. We present a novel recurrent neural network model that is capable of extracting information from an image or video by adaptively selecting a sequence of regions or locations and only processing the selected regions at high resolution. Like convolutional neural networks, the proposed model has a degree of translation invariance built-in, but the amount of computation it performs can be controlled independently of the input image size. While the model is non-differentiable, it can be trained using reinforcement learning methods to learn task-specific policies. We evaluate our model on several image classification tasks, where it significantly outperforms a convolutional neural network baseline on cluttered images, and on a dynamic visual control problem, where it learns to track a simple object without an explicit training signal for doing so.
* [[Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719)]
    [[pdf](https://arxiv.org/pdf/1703.02719.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.02719/)]
    * Title: Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
    * Year: 08 Mar `2017`
    * Authors: Chao Peng, Xiangyu Zhang, Gang Yu, Guiming Luo, Jian Sun
    * Abstract: One of recent trends [30, 31, 14] in network architec- ture design is stacking small filters (e.g., 1x1 or 3x3) in the entire network because the stacked small filters is more ef- ficient than a large kernel, given the same computational complexity. However, in the field of semantic segmenta- tion, where we need to perform dense per-pixel prediction, we find that the large kernel (and effective receptive field) plays an important role when we have to perform the clas- sification and localization tasks simultaneously. Following our design principle, we propose a Global Convolutional Network to address both the classification and localization issues for the semantic segmentation. We also suggest a residual-based boundary refinement to further refine the ob- ject boundaries. Our approach achieves state-of-art perfor- mance on two public benchmarks and significantly outper- forms previous results, 82.2% (vs 80.2%) on PASCAL VOC 2012 dataset and 76.9% (vs 71.8%) on Cityscapes dataset.
* [[MPViT: Multi-Path Vision Transformer for Dense Prediction](https://arxiv.org/abs/2112.11010)]
    [[pdf](https://arxiv.org/pdf/2112.11010.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2112.11010/)]
    * Title: MPViT: Multi-Path Vision Transformer for Dense Prediction
    * Year: 21 Dec `2021`
    * Authors: Youngwan Lee, Jonghee Kim, Jeff Willette, Sung Ju Hwang
    * Abstract: Dense computer vision tasks such as object detection and segmentation require effective multi-scale feature representation for detecting or classifying objects or regions with varying sizes. While Convolutional Neural Networks (CNNs) have been the dominant architectures for such tasks, recently introduced Vision Transformers (ViTs) aim to replace them as a backbone. Similar to CNNs, ViTs build a simple multi-stage structure (i.e., fine-to-coarse) for multi-scale representation with single-scale patches. In this work, with a different perspective from existing Transformers, we explore multi-scale patch embedding and multi-path structure, constructing the Multi-Path Vision Transformer (MPViT). MPViT embeds features of the same size~(i.e., sequence length) with patches of different scales simultaneously by using overlapping convolutional patch embedding. Tokens of different scales are then independently fed into the Transformer encoders via multiple paths and the resulting features are aggregated, enabling both fine and coarse feature representations at the same feature level. Thanks to the diverse, multi-scale feature representations, our MPViTs scaling from tiny~(5M) to base~(73M) consistently achieve superior performance over state-of-the-art Vision Transformers on ImageNet classification, object detection, instance segmentation, and semantic segmentation. These extensive results demonstrate that MPViT can serve as a versatile backbone network for various vision tasks. Code will be made publicly available at \url{this https URL}.
* [[Semantic Segmentation with Boundary Neural Fields](https://arxiv.org/abs/1511.02674)]
    [[pdf](https://arxiv.org/pdf/1511.02674.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.02674/)]
    * Title: Semantic Segmentation with Boundary Neural Fields
    * Year: 09 Nov `2015`
    * Authors: Gedas Bertasius, Jianbo Shi, Lorenzo Torresani
    * Abstract: The state-of-the-art in semantic segmentation is currently represented by fully convolutional networks (FCNs). However, FCNs use large receptive fields and many pooling layers, both of which cause blurring and low spatial resolution in the deep layers. As a result FCNs tend to produce segmentations that are poorly localized around object boundaries. Prior work has attempted to address this issue in post-processing steps, for example using a color-based CRF on top of the FCN predictions. However, these approaches require additional parameters and low-level features that are difficult to tune and integrate into the original network architecture. Additionally, most CRFs use color-based pixel affinities, which are not well suited for semantic segmentation and lead to spatially disjoint predictions. To overcome these problems, we introduce a Boundary Neural Field (BNF), which is a global energy model integrating FCN predictions with boundary cues. The boundary information is used to enhance semantic segment coherence and to improve object localization. Specifically, we first show that the convolutional filters of semantic FCNs provide good features for boundary detection. We then employ the predicted boundaries to define pairwise potentials in our energy. Finally, we show that our energy decomposes semantic segmentation into multiple binary problems, which can be relaxed for efficient global optimization. We report extensive experiments demonstrating that minimization of our global boundary-based energy yields results superior to prior globalization methods, both quantitatively as well as qualitatively.
* [[PCT: Point cloud transformer](https://arxiv.org/abs/2012.09688)]
    [[pdf](https://arxiv.org/pdf/2012.09688.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2012.09688/)]
    * Title: PCT: Point cloud transformer
    * Year: 17 Dec `2020`
    * Authors: Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R. Martin, Shi-Min Hu
    * Abstract: The irregular domain and lack of ordering make it challenging to design deep neural networks for point cloud processing. This paper presents a novel framework named Point Cloud Transformer(PCT) for point cloud learning. PCT is based on Transformer, which achieves huge success in natural language processing and displays great potential in image processing. It is inherently permutation invariant for processing a sequence of points, making it well-suited for point cloud learning. To better capture local context within the point cloud, we enhance input embedding with the support of farthest point sampling and nearest neighbor search. Extensive experiments demonstrate that the PCT achieves the state-of-the-art performance on shape classification, part segmentation and normal estimation tasks.
* [[Joint Semantic Segmentation and Boundary Detection using Iterative Pyramid Contexts](https://arxiv.org/abs/2004.07684)]
    [[pdf](https://arxiv.org/pdf/2004.07684.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2004.07684/)]
    * Title: Joint Semantic Segmentation and Boundary Detection using Iterative Pyramid Contexts
    * Year: 16 Apr `2020`
    * Authors: Mingmin Zhen, Jinglu Wang, Lei Zhou, Shiwei Li, Tianwei Shen, Jiaxiang Shang, Tian Fang, Quan Long
    * Abstract: In this paper, we present a joint multi-task learning framework for semantic segmentation and boundary detection. The critical component in the framework is the iterative pyramid context module (PCM), which couples two tasks and stores the shared latent semantics to interact between the two tasks. For semantic boundary detection, we propose the novel spatial gradient fusion to suppress nonsemantic edges. As semantic boundary detection is the dual task of semantic segmentation, we introduce a loss function with boundary consistency constraint to improve the boundary pixel accuracy for semantic segmentation. Our extensive experiments demonstrate superior performance over state-of-the-art works, not only in semantic segmentation but also in semantic boundary detection. In particular, a mean IoU score of 81:8% on Cityscapes test set is achieved without using coarse data or any external data for semantic segmentation. For semantic boundary detection, we improve over previous state-of-the-art works by 9.9% in terms of AP and 6:8% in terms of MF(ODS).
* [[FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting](https://arxiv.org/abs/2109.02974)]
    [[pdf](https://arxiv.org/pdf/2109.02974.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2109.02974/)]
    * Title: FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting
    * Year: 07 Sep `2021`
    * Authors: Rui Liu, Hanming Deng, Yangyi Huang, Xiaoyu Shi, Lewei Lu, Wenxiu Sun, Xiaogang Wang, Jifeng Dai, Hongsheng Li
    * Abstract: Transformer, as a strong and flexible architecture for modelling long-range relations, has been widely explored in vision tasks. However, when used in video inpainting that requires fine-grained representation, existed method still suffers from yielding blurry edges in detail due to the hard patch splitting. Here we aim to tackle this problem by proposing FuseFormer, a Transformer model designed for video inpainting via fine-grained feature fusion based on novel Soft Split and Soft Composition operations. The soft split divides feature map into many patches with given overlapping interval. On the contrary, the soft composition operates by stitching different patches into a whole feature map where pixels in overlapping regions are summed up. These two modules are first used in tokenization before Transformer layers and de-tokenization after Transformer layers, for effective mapping between tokens and features. Therefore, sub-patch level information interaction is enabled for more effective feature propagation between neighboring patches, resulting in synthesizing vivid content for hole regions in videos. Moreover, in FuseFormer, we elaborately insert the soft composition and soft split into the feed-forward network, enabling the 1D linear layers to have the capability of modelling 2D structure. And, the sub-patch level feature fusion ability is further enhanced. In both quantitative and qualitative evaluations, our proposed FuseFormer surpasses state-of-the-art methods. We also conduct detailed analysis to examine its superiority.

### 5.2. (2015, FCN) Patchwise Training (count=5)

* [[Toward automatic phenotyping of developing embryos from videos](https://ieeexplore.ieee.org/document/1495508)]
    * Title: Toward automatic phenotyping of developing embryos from videos
    * Year: `2005`
    * Author: Feng Ning
    * Abstract: We describe a trainable system for analyzing videos of developing C. elegans embryos. The system automatically detects, segments, and locates cells and nuclei in microscopic images. The system was designed as the central component of a fully automated phenotyping system. The system contains three modules 1) a convolutional network trained to classify each pixel into five categories: cell wall, cytoplasm, nucleus membrane, nucleus, outside medium; 2) an energy-based model, which cleans up the output of the convolutional network by learning local consistency constraints that must be satisfied by label images; 3) a set of elastic models of the embryo at various stages of development that are matched to the label images.
* [[Deep neural networks segment neuronal membranes in electron microscopy images](https://papers.nips.cc/paper/2012/hash/459a4ddcb586f24efd9395aa7662bc7c-Abstract.html)]
    [[pdf](https://proceedings.neurips.cc/paper/2012/file/459a4ddcb586f24efd9395aa7662bc7c-Paper.pdf)]
    * Title: Deep neural networks segment neuronal membranes in electron microscopy images
    * Year: 03 Dec `2012`
    * Authors: Dan Ciresan, Alessandro Giusti, Luca Gambardella, Jürgen Schmidhuber
    * Abstract: We address a central problem of neuroanatomy, namely, the automatic segmentation of neuronal structures depicted in stacks of electron microscopy (EM) images. This is necessary to efficiently map 3D brain structure and connectivity. To segment biological neuron membranes, we use a special type of deep artificial neural network as a pixel classifier. The label of each pixel (membrane or nonmembrane) is predicted from raw pixel values in a square window centered on it. The input layer maps each window pixel to a neuron. It is followed by a succession of convolutional and max-pooling layers which preserve 2D information and extract features with increasing levels of abstraction. The output layer produces a calibrated probability for each class. The classifier is trained by plain gradient descent on a 512 × 512 × 30 stack with known ground truth, and tested on a stack of the same size (ground truth unknown to the authors) by the organizers of the ISBI 2012 EM Segmentation Challenge. Even without problem-specific postprocessing, our approach outperforms competing techniques by a large margin in all three considered metrics, i.e. rand error, warping error and pixel error. For pixel error, our approach is the only one outperforming a second human observer.
    * Comments:
        * > Thousands of training images are usually beyond reach in biomedical tasks. Hence, Ciresan et al. [1] trained a network in a sliding-window setup to predict the class label of each pixel by providing a local region (patch) around that pixel as input. (U-Net, 2015)
* [[Learning Hierarchical Features for Scene Labeling](https://ieeexplore.ieee.org/document/6338939)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6338939)]
    * Title: Learning Hierarchical Features for Scene Labeling
    * Year: `2013`
    * Authors: Clement Farabet; Camille Couprie; Laurent Najman; Yann LeCun
    * Abstract: Scene labeling consists of labeling each pixel in an image with the category of the object it belongs to. We propose a method that uses a multiscale convolutional network trained from raw pixels to extract dense feature vectors that encode regions of multiple sizes centered on each pixel. The method alleviates the need for engineered features, and produces a powerful representation that captures texture, shape, and contextual information. We report results using multiple postprocessing methods to produce the final labeling. Among those, we propose a technique to automatically retrieve, from a pool of segmentation components, an optimal set of components that best explain the scene; these components are arbitrary, for example, they can be taken from a segmentation tree or from any family of oversegmentations. The system yields record accuracies on the SIFT Flow dataset (33 classes) and the Barcelona dataset (170 classes) and near-record accuracy on Stanford background dataset (eight classes), while being an order of magnitude faster than competing approaches, producing a 320×240 image labeling in less than a second, including feature extraction.
    * Comments:
        * > Among the first have been Farabet et al. (2013) who apply DCNNs at multiple image resolutions and then employ a segmentation tree to smooth the prediction results. (DeepLabv1, 2014)
        * > Farabet et al. (2013) treat superpixels as nodes for a local pairwise CRF and use graph-cuts for discrete inference. (DeepLabv1, 2014)
        * > Farabet et al. [19] employed a Laplacian pyramid, passed each scale through a shared network, and fused the features from all the scales. (Attention to Scale, 2015)
        * > color based segmentation (ENet, 2016)
* [[Recurrent Convolutional Neural Networks for Scene Labeling](https://proceedings.mlr.press/v32/pinheiro14.html)]
    [[pdf](http://proceedings.mlr.press/v32/pinheiro14.pdf)]
    * Title: Recurrent Convolutional Neural Networks for Scene Labeling
    * Year: Jun `2014`
    * Authors: Pedro Pinheiro, Ronan Collobert
    * Institutions: [Ecole Polytechnique Fed´ erale de Lausanne (EPFL), Lausanne, Switzerland], [Idiap Research Institute, Martigny, Switzerland]
    * Abstract: The goal of the scene labeling task is to assign a class label to each pixel in an image. To ensure a good visual coherence and a high class accuracy, it is essential for a model to capture long range pixel) label dependencies in images. In a feed-forward architecture, this can be achieved simply by considering a sufficiently large input context patch, around each pixel to be labeled. We propose an approach that consists of a recurrent convolutional neural network which allows us to consider a large input context while limiting the capacity of the model. Contrary to most standard approaches, our method does not rely on any segmentation technique nor any task-specific features. The system is trained in an end-to-end manner over raw pixels, and models complex spatial dependencies with low inference cost. As the context size increases with the built-in recurrence, the system identifies and corrects its own errors. Our approach yields state-of-the-art performance on both the Stanford Background Dataset and the SIFT Flow Dataset, while remaining very fast at test time.
    * Comments:
        * > Pinheiro et al. [45], instead of applying multi-scale input images at once, fed multi-scale images at different stages in a recurrent convolutional neural network. (Attention to Scale, 2015)
* [[$N^4$-Fields](https://arxiv.org/abs/1406.6558)]
    [[pdf](https://arxiv.org/pdf/1406.6558.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1406.6558/)]
    * Title: $N^4$-Fields: Neural Network Nearest Neighbor Fields for Image Transforms
    * Year: 25 Jun `2014`
    * Authors: Yaroslav Ganin, Victor Lempitsky
    * Abstract: We propose a new architecture for difficult image processing operations, such as natural edge detection or thin object segmentation. The architecture is based on a simple combination of convolutional neural networks with the nearest neighbor search. We focus our attention on the situations when the desired image transformation is too hard for a neural network to learn explicitly. We show that in such situations, the use of the nearest neighbor search on top of the network output allows to improve the results considerably and to account for the underfitting effect during the neural network training. The approach is validated on three challenging benchmarks, where the performance of the proposed architecture matches or exceeds the state-of-the-art.
* [[Are spatial and global constraints really necessary for segmentation?](https://ieeexplore.ieee.org/document/6126219)]
    * Title: Are spatial and global constraints really necessary for segmentation?

### 5.3. Fully Convolutional Networks (FCN)

> Semantic segmentation datasets have a rich history [35, 23, 9] and helped drive key innovations (e.g., fully convolutional nets [26] were developed using [23, 9]). (Panoptic Segmentation, 2018)

* [[SIFT Flow](https://ieeexplore.ieee.org/document/5551153)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5551153)]
    * Title: SIFT Flow: Dense Correspondence across Scenes and Its Applications
    * Year: 19 August `2010`
    * Authors: Ce Liu; Jenny Yuen; Antonio Torralba
    * Abstract: While image alignment has been studied in different areas of computer vision for decades, aligning images depicting different scenes remains a challenging problem. Analogous to optical flow, where an image is aligned to its temporally adjacent frame, we propose SIFT flow, a method to align an image to its nearest neighbors in a large image corpus containing a variety of scenes. The SIFT flow algorithm consists of matching densely sampled, pixelwise SIFT features between two images while preserving spatial discontinuities. The SIFT features allow robust matching across different scene/object appearances, whereas the discontinuity-preserving spatial model allows matching of objects located at different parts of the scene. Experiments show that the proposed approach robustly aligns complex scene pairs containing significant spatial differences. Based on SIFT flow, we propose an alignment-based large database framework for image analysis and synthesis, where image information is transferred from the nearest neighbors to a query image according to the dense scene correspondence. This framework is demonstrated through concrete applications such as motion field prediction from a single image, motion synthesis via object transfer, satellite image registration, and face recognition.
* [[Fully Convolutional Networks (FCN)](https://arxiv.org/abs/1411.4038)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1411.4038.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1411.4038/)]
    * Title: Fully Convolutional Networks for Semantic Segmentation
    * Year: 14 Nov `2014`
    * Authors: Jonathan Long, Evan Shelhamer, Trevor Darrell
    * Institutions: [UC Berkeley]
    * Abstract: Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build "fully convolutional" networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet, the VGG net, and GoogLeNet) into fully convolutional networks and transfer their learned representations by fine-tuning to the segmentation task. We then define a novel architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional network achieves state-of-the-art segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes one third of a second for a typical image.
    * Comments:
        * > (2014, DeepLabV1) More recently, the segmentation-free techniques of (Long et al., 2014; Eigen & Fergus, 2014) directly apply DCNNs to the whole image in a sliding window fashion, replacing the last fully connected layers of a DCNN by convolutional layers. In order to deal with the spatial localization issues outlined in the beginning of the introduction, Long et al. (2014) upsample and concatenate the scores from inter-mediate feature maps, while Eigen & Fergus (2014) refine the prediction result from coarse to fine by propagating the coarse results to another DCNN.
        * > (2015, Dilated Convolutions) Long et al. (2015) showed that convolutional network architectures that had originally been developed for image classification can be successfully repurposed for dense prediction.
        * > (2015, Dilated Convolutions) In recent work on convolutional networks for semantic segmentation, Long et al. (2015) analyzed filter dilation but chose not to use it.
        * > (2015, DeconvNet) The main advantage of the methods based on FCN is that the network accepts a whole image as an input and performs fast and accurate inference.
        * > (2015, DeconvNet) Semantic segmentation based on FCNs [1, 17] have a couple of critical limitations. First, the network can handle only a single scale semantics within image due to the fixed-size receptive field. Therefore, the object that is substantially larger or smaller than the receptive field may be fragmented or mislabeled. ... Second, the detailed structures of an object are often lost or smoothed because the label map, input to the deconvolutional layer, is too coarse and deconvolution procedure is overly simple.
        * > (2015, DeconvNet) In this approach, fully connected layers in the standard CNNs are interpreted as convolutions with large receptive fields, and segmentation is achieved using coarse class score maps obtained by feedforwarding an input image. An interesting idea in this work is that a simple interpolation filter is employed for deconvolution and only the CNN part of the network is fine-tuned to learn deconvolution indirectly.
        * > (2015, Attention to Scale) FCN-8s [38] gradually learns finer-scale prediction from lower layers (initialized with coarser-scale prediction).
        * > (2015, U-Net) The main idea in [9] is to supplement a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators. Hence, these layers increase the resolution of the output. In order to localize, high resolution features from the contracting path are combined with the upsampled output. A successive convolution layer can then learn to assemble a more precise output based on this information.
        * > (2015, SegNet) Each decoder in the Fully Convolutional Network (FCN) architecture [2] learns to upsample its input feature map(s) and combines them with the corresponding encoder feature map to produce the input to the next decoder.
        * > (2015, ParseNet) The FCN approach can be thought of as sliding an classification network around an input image, and processes each sliding window area independently. In particular, FCN disregards global information about an image, thus ignoring potentially useful scene-level semantic context.
        * > (2016, FPN) FCN [24] sums partial scores for each category over multiple scales to compute semantic segmentations.
        * > (2016, RefineNet) The FCN method in [36] adds prediction layers to middle layers to generate prediction scores at multiple resolutions. They average the multi-resolution scores to generate the final prediction mask.
        * > (2021, PVT) In the early stages, FCN introduced a fully convolutional architecture to generate a spatial segmentation map for a given image of any size.
        * skip connections??? where?

### 5.4. FCN based models (Attention to Scale, 2015) (6)

* DeepLabV1
* BoxSup
* Semantic image segmentation via deep parsing network
* Learning Deconvolution Network for Semantic Segmentation
* Conditional Random Fields as Recurrent Neural Networks
* Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation

### 5.5. The U-Net Family

* [[U-Net](https://arxiv.org/abs/1505.04597)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1505.04597.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1505.04597/)]
    * Title: U-Net: Convolutional Networks for Biomedical Image Segmentation
    * Year: 18 May `2015`
    * Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    * Institutions: [Computer Science Department and BIOSS Centre for Biological Signalling Studies, University of Freiburg, Germany]
    * Abstract: There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at this http URL .
    * Comments:
        * > (2015, SegNet) As compared to SegNet, U-Net [16] (proposed for the medical imaging community) does not reuse pooling indices but instead transfers the entire feature map (at the cost of more memory) to the corresponding decoders and concatenates them to upsampled (via deconvolution) decoder feature maps.
        * > (2016, RefineNet) The method Seg-Net [2] and U-Net [40] apply skip-connections in the deconvolution architecture to exploit the features from middle layers.
        * > (2021, PVT) Inspired by FCN, U-Net [37] is proposed for especially the medical image segmentation domain, which bridges the information flow between corresponding low-level and high-level feature maps with the same spatial sizes.
* [[UNet++](https://arxiv.org/abs/1807.10165)]
    [[pdf](https://arxiv.org/pdf/1807.10165.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1807.10165/)]
    * Title: UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    * Year: 18 Jul `2018`
    * Authors: Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang
    * Abstract: In this paper, we present UNet++, a new, more powerful architecture for medical image segmentation. Our architecture is essentially a deeply-supervised encoder-decoder network where the encoder and decoder sub-networks are connected through a series of nested, dense skip pathways. The re-designed skip pathways aim at reducing the semantic gap between the feature maps of the encoder and decoder sub-networks. We argue that the optimizer would deal with an easier learning task when the feature maps from the decoder and encoder networks are semantically similar. We have evaluated UNet++ in comparison with U-Net and wide U-Net architectures across multiple medical image segmentation tasks: nodule segmentation in the low-dose CT scans of chest, nuclei segmentation in the microscopy images, liver segmentation in abdominal CT scans, and polyp segmentation in colonoscopy videos. Our experiments demonstrate that UNet++ with deep supervision achieves an average IoU gain of 3.9 and 3.4 points over U-Net and wide U-Net, respectively.
* [[Sharp U-Net](https://arxiv.org/abs/2107.12461)]
    [[pdf](https://arxiv.org/pdf/2107.12461.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2107.12461/)]
    * Title: Sharp U-Net: Depthwise Convolutional Network for Biomedical Image Segmentation
    * Year: 26 Jul `2021`
    * Authors: Hasib Zunair, A. Ben Hamza
    * Abstract: The U-Net architecture, built upon the fully convolutional network, has proven to be effective in biomedical image segmentation. However, U-Net applies skip connections to merge semantically different low- and high-level convolutional features, resulting in not only blurred feature maps, but also over- and under-segmented target regions. To address these limitations, we propose a simple, yet effective end-to-end depthwise encoder-decoder fully convolutional network architecture, called Sharp U-Net, for binary and multi-class biomedical image segmentation. The key rationale of Sharp U-Net is that instead of applying a plain skip connection, a depthwise convolution of the encoder feature map with a sharpening kernel filter is employed prior to merging the encoder and decoder features, thereby producing a sharpened intermediate feature map of the same size as the encoder map. Using this sharpening filter layer, we are able to not only fuse semantically less dissimilar features, but also to smooth out artifacts throughout the network layers during the early stages of training. Our extensive experiments on six datasets show that the proposed Sharp U-Net model consistently outperforms or matches the recent state-of-the-art baselines in both binary and multi-class segmentation tasks, while adding no extra learnable parameters. Furthermore, Sharp U-Net outperforms baselines that have more than three times the number of learnable parameters.
* [[SMU-Net](https://arxiv.org/abs/2204.02961)]
    [[pdf](https://arxiv.org/pdf/2204.02961.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2204.02961/)]
    * Title: SMU-Net: Style matching U-Net for brain tumor segmentation with missing modalities
    * Year: 06 Apr `2022`
    * Authors: Reza Azad, Nika Khosravi, Dorit Merhof
    * Abstract: Gliomas are one of the most prevalent types of primary brain tumours, accounting for more than 30\% of all cases and they develop from the glial stem or progenitor cells. In theory, the majority of brain tumours could well be identified exclusively by the use of Magnetic Resonance Imaging (MRI). Each MRI modality delivers distinct information on the soft tissue of the human brain and integrating all of them would provide comprehensive data for the accurate segmentation of the glioma, which is crucial for the patient's prognosis, diagnosis, and determining the best follow-up treatment. Unfortunately, MRI is prone to artifacts for a variety of reasons, which might result in missing one or more MRI modalities. Various strategies have been proposed over the years to synthesize the missing modality or compensate for the influence it has on automated segmentation models. However, these methods usually fail to model the underlying missing information. In this paper, we propose a style matching U-Net (SMU-Net) for brain tumour segmentation on MRI images. Our co-training approach utilizes a content and style-matching mechanism to distill the informative features from the full-modality network into a missing modality network. To do so, we encode both full-modality and missing-modality data into a latent space, then we decompose the representation space into a style and content representation. Our style matching module adaptively recalibrates the representation space by learning a matching function to transfer the informative and textural features from a full-modality path into a missing-modality path. Moreover, by modelling the mutual information, our content module surpasses the less informative features and re-calibrates the representation space based on discriminative semantic features. The evaluation process on the BraTS 2018 dataset shows a significant results.

### 5.6. Encoder-Decoder Architectures

> (2015, SegNet) Newer deep architectures [2, 4, 13, 18, 10] particularly designed for segmentation have advanced the state-of-the-art by learning to decode or map low resolution image representations to pixel-wise predictions.

* [[DeconvNet](https://arxiv.org/abs/1505.04366)]
    [[pdf](https://arxiv.org/pdf/1505.04366.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1505.04366/)]
    * Title: Learning Deconvolution Network for Semantic Segmentation
    * Year: 17 May `2015`
    * Authors: Hyeonwoo Noh, Seunghoon Hong, Bohyung Han
    * Institutions: [Department of Computer Science and Engineering, POSTECH, Korea]
    * Abstract: We propose a novel semantic segmentation algorithm by learning a deconvolution network. We learn the network on top of the convolutional layers adopted from VGG 16-layer net. The deconvolution network is composed of deconvolution and unpooling layers, which identify pixel-wise class labels and predict segmentation masks. We apply the trained network to each proposal in an input image, and construct the final semantic segmentation map by combining the results from all proposals in a simple manner. The proposed algorithm mitigates the limitations of the existing methods based on fully convolutional networks by integrating deep deconvolution network and proposal-wise prediction; our segmentation method typically identifies detailed structures and handles objects in multiple scales naturally. Our network demonstrates outstanding performance in PASCAL VOC 2012 dataset, and we achieve the best accuracy (72.5%) among the methods trained with no external data through ensemble with the fully convolutional network.
    * Comments:
        * > Recent work has studied two approaches to dealing with the conflicting demands of multi-scale reasoning and full-resolution dense prediction. One approach involves repeated up-convolutions that aim to recover lost resolution while carrying over the global perspective from downsampled layers (Noh et al., 2015; Fischer et al., 2015). (Dilated Convolutions, 2015)
        * > (2015, SegNet) The recently proposed Deconvolutional Network [4] and its semi-supervised variant the Decoupled network [18] use the max locations of the encoder feature maps (pooling indices) to perform non-linear upsampling in the decoder network.
        * > (2015, SegNet) DeconvNet has a much larger parameterization, needs more computational resources and is harder to train end-to-end (Table VI), primarily due to the use of fully connected layers (albeit in a convolutional manner).
        * > (2016, PSPNet) Noh et al. [30] proposed a coarse-to-fine structure with deconvolution network to learn the segmentation mask.
* [[DecoupledNet](https://arxiv.org/abs/1506.04924)]
    [[pdf](https://arxiv.org/pdf/1506.04924.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1506.04924/)]
    * Title: Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation
    * Year: 16 Jun `2015`
    * Authors: Seunghoon Hong, Hyeonwoo Noh, Bohyung Han
    * Abstract: We propose a novel deep neural network architecture for semi-supervised semantic segmentation using heterogeneous annotations. Contrary to existing approaches posing semantic segmentation as a single task of region-based classification, our algorithm decouples classification and segmentation, and learns a separate network for each task. In this architecture, labels associated with an image are identified by classification network, and binary segmentation is subsequently performed for each identified label in segmentation network. The decoupled architecture enables us to learn classification and segmentation networks separately based on the training data with image-level and pixel-wise class labels, respectively. It facilitates to reduce search space for segmentation effectively by exploiting class-specific activation maps obtained from bridging layers. Our algorithm shows outstanding performance compared to other semi-supervised approaches even with much less training images with strong annotations in PASCAL VOC dataset.
    * Comments:
        * Semi-supervised variant of the DeconvNet. (2015, SegNet)
        * > The recently proposed Deconvolutional Network [4] and its semi-supervised variant the Decoupled network [18] use the max locations of the encoder feature maps (pooling indices) to perform non-linear upsampling in the decoder network. (2015, SegNet)
* [[SegNet-Basic](https://arxiv.org/abs/1505.07293)]
    [[pdf](https://arxiv.org/pdf/1505.07293.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1505.07293/)]
    * Title: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling
    * Year: 27 May `2015`
    * Authors: Vijay Badrinarayanan, Ankur Handa, Roberto Cipolla
    * Institutions: [Machine Intelligence Lab, Department of Engineering, University of Cambridge, UK]
    * Abstract: We propose a novel deep architecture, SegNet, for semantic pixel wise image labelling. SegNet has several attractive properties; (i) it only requires forward evaluation of a fully learnt function to obtain smooth label predictions, (ii) with increasing depth, a larger context is considered for pixel labelling which improves accuracy, and (iii) it is easy to visualise the effect of feature activation(s) in the pixel label space at any depth. SegNet is composed of a stack of encoders followed by a corresponding decoder stack which feeds into a soft-max classification layer. The decoders help map low resolution feature maps at the output of the encoder stack to full input image size feature maps. This addresses an important drawback of recent deep learning approaches which have adopted networks designed for object categorization for pixel wise labelling. These methods lack a mechanism to map deep layer feature maps to input dimensions. They resort to ad hoc methods to upsample features, e.g. by replication. This results in noisy predictions and also restricts the number of pooling layers in order to avoid too much upsampling and thus reduces spatial context. SegNet overcomes these problems by learning to map encoder outputs to image pixel labels. We test the performance of SegNet on outdoor RGB scenes from CamVid, KITTI and indoor scenes from the NYU dataset. Our results show that SegNet achieves state-of-the-art performance even without use of additional cues such as depth, video frames or post-processing with CRF models.
* [[SegNet](https://arxiv.org/abs/1511.00561)]
    [[pdf](https://arxiv.org/pdf/1511.00561.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.00561/)]
    * Title: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
    * Year: 02 Nov `2015`
    * Authors: Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
    * Institutions: [Machine Intelligence Lab, Department of Engineering, University of Cambridge, UK]
    * Abstract: We present a novel and practical deep fully convolutional neural network architecture for semantic pixel-wise segmentation termed SegNet. This core trainable segmentation engine consists of an encoder network, a corresponding decoder network followed by a pixel-wise classification layer. The architecture of the encoder network is topologically identical to the 13 convolutional layers in the VGG16 network. The role of the decoder network is to map the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification. The novelty of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature map(s). Specifically, the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling. This eliminates the need for learning to upsample. The upsampled maps are sparse and are then convolved with trainable filters to produce dense feature maps. We compare our proposed architecture with the widely adopted FCN and also with the well known DeepLab-LargeFOV, DeconvNet architectures. This comparison reveals the memory versus accuracy trade-off involved in achieving good segmentation performance. SegNet was primarily motivated by scene understanding applications. Hence, it is designed to be efficient both in terms of memory and computational time during inference. It is also significantly smaller in the number of trainable parameters than other competing architectures. We also performed a controlled benchmark of SegNet and other architectures on both road scenes and SUN RGB-D indoor scene segmentation tasks. We show that SegNet provides good performance with competitive inference time and more efficient inference memory-wise as compared to other architectures. We also provide a Caffe implementation of SegNet and a web demo at this http URL.
    * Comments:
        * > (2016, ENet) Inspired by probabilistic auto-encoders ranzato07 ; ngiam11 , encoder-decoder network architecture has been introduced in SegNet-basic badrinarayanan15basic , and further improved in SegNet badrinarayanan15 .
        * > (2016, ENet) SegNet is a very symmetric architecture, as the encoder is an exact mirror of the encoder.
        * > (2016, RefineNet) The method Seg-Net [2] and U-Net [40] apply skip-connections in the deconvolution architecture to exploit the features from middle layers.
* [[SegNeXt](https://arxiv.org/abs/2209.08575)]
    [[pdf](https://arxiv.org/pdf/2209.08575.pdf)]
    [vanity]
    * Title: SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation
    * Year: 18 Sep `2022`
    * Authors: Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zhengning Liu, Ming-Ming Cheng, Shi-Min Hu
    * Institutions: [Tsinghua University], [Nankai University], [Fitten Tech, Beijing, China]
    * Abstract: We present SegNeXt, a simple convolutional network architecture for semantic segmentation. Recent transformer-based models have dominated the field of semantic segmentation due to the efficiency of self-attention in encoding spatial information. In this paper, we show that convolutional attention is a more efficient and effective way to encode contextual information than the self-attention mechanism in transformers. By re-examining the characteristics owned by successful segmentation models, we discover several key components leading to the performance improvement of segmentation models. This motivates us to design a novel convolutional attention network that uses cheap convolutional operations. Without bells and whistles, our SegNeXt significantly improves the performance of previous state-of-the-art methods on popular benchmarks, including ADE20K, Cityscapes, COCO-Stuff, Pascal VOC, Pascal Context, and iSAID. Notably, SegNeXt outperforms EfficientNet-L2 w/ NAS-FPN and achieves 90.6% mIoU on the Pascal VOC 2012 test leaderboard using only 1/10 parameters of it. On average, SegNeXt achieves about 2.0% mIoU improvements compared to the state-of-the-art methods on the ADE20K datasets with the same or fewer computations. Code is available at this https URL (Jittor) and this https URL (Pytorch).
* [[ENet](https://arxiv.org/abs/1606.02147)]
    [[pdf](https://arxiv.org/pdf/1606.02147.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1606.02147/)]
    * Title: ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
    * Year: 07 Jun `2016`
    * Authors: Adam Paszke, Abhishek Chaurasia, Sangpil Kim, Eugenio Culurciello
    * Institutions: [Faculty of Mathematics, Informatics and Mechanics, University of Warsaw, Poland], [Electrical and Computer Engineering, Purdue University, USA]
    * Abstract: The ability to perform pixel-wise semantic segmentation in real-time is of paramount importance in mobile applications. Recent deep neural networks aimed at this task have the disadvantage of requiring a large number of floating point operations and have long run-times that hinder their usability. In this paper, we propose a novel deep neural network architecture named ENet (efficient neural network), created specifically for tasks requiring low latency operation. ENet is up to 18$\times$ faster, requires 75$\times$ less FLOPs, has 79$\times$ less parameters, and provides similar or better accuracy to existing models. We have tested it on CamVid, Cityscapes and SUN datasets and report on comparisons with existing state-of-the-art methods, and the trade-offs between accuracy and processing time of a network. We present performance measurements of the proposed architecture on embedded systems and suggest possible software improvements that could make ENet even faster.
    * Comments:
        * > The recent ENet [11] sits on the opposite side in terms of efficiency, in which authors also adapt ResNet to the segmentation task, but make important sacrifices in the network layers to gain efficiency at the expense of a lower accuracy compared to the other approaches. (2017, ERFNet)
* [[Efficient ConvNet for real-time semantic segmentation](https://ieeexplore.ieee.org/document/7995966)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7995966)]
    * Title: Efficient ConvNet for real-time semantic segmentation
    * Year: 31 July `2017`
    * Authors: Eduardo Romera; José M. Álvarez; Luis M. Bergasa; Roberto Arroyo
    * Institutions: [Electronics Department, University of Alcalá (UAH), Alcalá de Henares, Spain], [CSIRO-Data61, Canberra, Australia]
    * Abstract: Semantic segmentation is a task that covers most of the perception needs of intelligent vehicles in an unified way. ConvNets excel at this task, as they can be trained end-to-end to accurately classify multiple object categories in an image at the pixel level. However, current approaches normally involve complex architectures that are expensive in terms of computational resources and are not feasible for ITS applications. In this paper, we propose a deep architecture that is able to run in real-time while providing accurate semantic segmentation. The core of our ConvNet is a novel layer that uses residual connections and factorized convolutions in order to remain highly efficient while still retaining remarkable performance. Our network is able to run at 83 FPS in a single Titan X, and at more than 7 FPS in a Jetson TX1 (embedded GPU). A comprehensive set of experiments demonstrates that our system, trained from scratch on the challenging Cityscapes dataset, achieves a classification performance that is among the state of the art, while being orders of magnitude faster to compute than other architectures that achieve top precision. This makes our model an ideal approach for scene understanding in intelligent vehicles applications.
* [[ERFNet](https://ieeexplore.ieee.org/document/8063438)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8063438)]
    * Title: ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation
    * Year: 09 Oct `2017`
    * Authors: Eduardo Romera; José M. Álvarez; Luis M. Bergasa; Roberto Arroyo
    * Institutions: [Department of Electronics, University of Alcalá, Alcalá de Henares, Spain], [CSIRO-Data61, Canberra, Australia]
    * Abstract: Semantic segmentation is a challenging task that addresses most of the perception needs of intelligent vehicles (IVs) in an unified way. Deep neural networks excel at this task, as they can be trained end-to-end to accurately classify multiple object categories in an image at pixel level. However, a good tradeoff between high quality and computational resources is yet not present in the state-of-the-art semantic segmentation approaches, limiting their application in real vehicles. In this paper, we propose a deep architecture that is able to run in real time while providing accurate semantic segmentation. The core of our architecture is a novel layer that uses residual connections and factorized convolutions in order to remain efficient while retaining remarkable accuracy. Our approach is able to run at over 83 FPS in a single Titan X, and 7 FPS in a Jetson TX1 (embedded device). A comprehensive set of experiments on the publicly available Cityscapes data set demonstrates that our system achieves an accuracy that is similar to the state of the art, while being orders of magnitude faster to compute than other architectures that achieve top precision. The resulting tradeoff makes our model an ideal approach for scene understanding in IV applications. The code is publicly available at: https://github.com/Eromera/erfnet.

### 5.7. Increase feature resolution (Panoptic FPN, 2019) (4)

> As an alternative to dilation, an encoder-decoder [2] or 'U-Net' [47] architecture can be used to increase feature resolution [25, 42, 19, 45]. (Panoptic FPN, 2019)

* [[Recombinator Networks (RCN)](https://arxiv.org/abs/1511.07356)]
    [[pdf](https://arxiv.org/pdf/1511.07356.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.07356/)]
    * Title: Recombinator Networks: Learning Coarse-to-Fine Feature Aggregation
    * Year: 23 Nov `2015`
    * Authors: Sina Honari, Jason Yosinski, Pascal Vincent, Christopher Pal
    * Institutions: [University of Montreal], [Cornell University], [Ecole Polytechnique of Montreal], [CIFAR]
    * Abstract: Deep neural networks with alternating convolutional, max-pooling and decimation layers are widely used in state of the art architectures for computer vision. Max-pooling purposefully discards precise spatial information in order to create features that are more robust, and typically organized as lower resolution spatial feature maps. On some tasks, such as whole-image classification, max-pooling derived features are well suited; however, for tasks requiring precise localization, such as pixel level prediction and segmentation, max-pooling destroys exactly the information required to perform well. Precise localization may be preserved by shallow convnets without pooling but at the expense of robustness. Can we have our max-pooled multi-layered cake and eat it too? Several papers have proposed summation and concatenation based methods for combining upsampled coarse, abstract features with finer features to produce robust pixel level predictions. Here we introduce another model --- dubbed Recombinator Networks --- where coarse features inform finer features early in their formation such that finer features can make use of several layers of computation in deciding how to use coarse features. The model is trained once, end-to-end and performs better than summation-based architectures, reducing the error from the previous state of the art on two facial keypoint datasets, AFW and AFLW, by 30\% and beating the current state-of-the-art on 300W without using extra data. We improve performance even further by adding a denoising prediction model based on a novel convnet formulation.
    * Comments:
        * > (2016, FPN) Similar architectures adopting top-down and skip connections are popular in recent research [28, 17, 8, 26]. Their goals are to produce a single high-level feature map of a fine resolution on which the predictions are to be made (Fig. 2 top).
* Stacked Hourglass Networks for Human Pose Estimation
* [[The Laplacian Pyramid as a Compact Image Code](https://ieeexplore.ieee.org/document/1095851)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1095851)]
    * Title: The Laplacian Pyramid as a Compact Image Code
    * Authors: P. Burt; E. Adelson
* [[Laplacian Pyramid Reconstruction and Refinement (LRR)](https://arxiv.org/abs/1605.02264)]
    [[pdf](https://arxiv.org/pdf/1605.02264.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1605.02264/)]
    * Title: Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation
    * Year: 08 May `2016`
    * Authors: Golnaz Ghiasi, Charless C. Fowlkes
    * Institutions: [Dept. of Computer Science, University of California, Irvine]
    * Abstract: CNN architectures have terrific recognition performance but rely on spatial pooling which makes it difficult to adapt them to tasks that require dense, pixel-accurate labeling. This paper makes two contributions: (1) We demonstrate that while the apparent spatial resolution of convolutional feature maps is low, the high-dimensional feature representation contains significant sub-pixel localization information. (2) We describe a multi-resolution reconstruction architecture based on a Laplacian pyramid that uses skip connections from higher resolution feature maps and multiplicative gating to successively refine segment boundaries reconstructed from lower-resolution maps. This approach yields state-of-the-art semantic segmentation results on the PASCAL VOC and Cityscapes segmentation benchmarks without resorting to more complex random-field inference or instance detection driven architectures.
    * Comments:
        * > Ghiasi and Fowlkes [19] (LRR) propose a complex architecture that constructs a Laplacian pyramid to process and combine features at multiple scales. (2017, ERFNet)
* Learning to Refine Object Segments (SharpMask)

### 5.8. Multi Scale Architectures (Attention to Scale, 2015) (1 + 4 + 1 + 4 + 1)

* [[Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375)]
    [[pdf](https://arxiv.org/pdf/1504.06375.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1504.06375/)]
    * Title: Holistically-Nested Edge Detection
    * Year: 24 Apr `2015`
    * Authors: Saining Xie, Zhuowen Tu
    * Abstract: We develop a new edge detection algorithm that tackles two important issues in this long-standing vision problem: (1) holistic image training and prediction; and (2) multi-scale and multi-level feature learning. Our proposed method, holistically-nested edge detection (HED), performs image-to-image prediction by means of a deep learning model that leverages fully convolutional neural networks and deeply-supervised nets. HED automatically learns rich hierarchical representations (guided by deep supervision on side responses) that are important in order to approach the human ability resolve the challenging ambiguity in edge and object boundary detection. We significantly advance the state-of-the-art on the BSD500 dataset (ODS F-score of .782) and the NYU Depth dataset (ODS F-score of .746), and do so with an improved speed (0.4 second per image) that is orders of magnitude faster than some recent CNN-based edge detection algorithms.

> The first type, which we refer to as skip-net, combines features from the intermediate layers of FCNs [27, 38, 41, 11]. Features within a skip-net are multi-scale in nature due to the increasingly large receptive field sizes. During training, a skip-net usually employs a two-step process [27, 38, 41, 11], where it first trains the deep network backbone and then fixes or slightly fine-tunes during multi-scale feature extraction. The problem with this strategy is that the training process is not ideal (i.e., classifier training and feature-extraction are separate) and the training time is usually long (e.g., three to five days [38]). (Attention to Scale, 2015)

* Hypercolumns for Object Segmentation and Fine-grained Localization
* Fully Convolutional Networks for Semantic Segmentation
* Feedforward semantic segmentation with zoom-out features
* Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs

> The first type, skip-net, exploits features from different levels of the network. (Attention to Scale, 2015)

* [[ParseNet](https://arxiv.org/abs/1506.04579)]
    [[pdf](https://arxiv.org/pdf/1506.04579.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1506.04579/)]
    * Title: ParseNet: Looking Wider to See Better
    * Year: 15 Jun `2015`
    * Authors: Wei Liu, Andrew Rabinovich, Alexander C. Berg
    * Abstract: We present a technique for adding global context to deep convolutional networks for semantic segmentation. The approach is simple, using the average feature for a layer to augment the features at each location. In addition, we study several idiosyncrasies of training, significantly increasing the performance of baseline networks (e.g. from FCN). When we add our proposed global feature, and a technique for learning normalization parameters, accuracy increases consistently even over our improved versions of the baselines. Our proposed approach, ParseNet, achieves state-of-the-art performance on SiftFlow and PASCAL-Context with small additional computational cost over baselines, and near current state-of-the-art performance on PASCAL VOC 2012 semantic segmentation with a simple approach. Code is available at this https URL .
    * Comments:
        * > (2015, Attention to Scale) ParseNet [36] aggregated features over the whole image to provide global contextual information.
        * FCN is the core of ParseNet (2015, SegNet)
        * > (2016, PSPNet) Liu et al. [24] proved that global average pooling with FCN can improve semantic segmentation results.
        * > (2016, FPN) Several other approaches (HyperNet [18], ParseNet [23], and ION [2]) concatenate features of multiple layers before computing predictions, which is equivalent to summing transformed features.

> The second type, which we refer to as share-net, resizes the input image to several scales and passes each through a shared deep network. It then computes the final prediction based on the fusion of the resulting multi-scale features [19, 34]. A share-net does not need the two-step training process mentioned above. It usually employs average- or max-pooling over scales [20, 14, 44, 15]. Features at each scale are either equally important or sparsely selected. (Attention to Scale, 2015)

* Learning Hierarchical Features for Scene Labeling
* Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation
* [[Untangling Local and Global Deformations in Deep Convolutional Networks for Image Classification and Sliding Window Detection](https://arxiv.org/abs/1412.0296)]
    [[pdf](https://arxiv.org/pdf/1412.0296.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.0296/)]
    * Title: Untangling Local and Global Deformations in Deep Convolutional Networks for Image Classification and Sliding Window Detection
    * Year: 30 Nov `2014`
    * Authors: George Papandreou, Iasonas Kokkinos, Pierre-André Savalle
    * Abstract: Deep Convolutional Neural Networks (DCNNs) commonly use generic 'max-pooling' (MP) layers to extract deformation-invariant features, but we argue in favor of a more refined treatment. First, we introduce epitomic convolution as a building block alternative to the common convolution-MP cascade of DCNNs; while having identical complexity to MP, Epitomic Convolution allows for parameter sharing across different filters, resulting in faster convergence and better generalization. Second, we introduce a Multiple Instance Learning approach to explicitly accommodate global translation and scaling when training a DCNN exclusively with class labels. For this we rely on a 'patchwork' data structure that efficiently lays out all image scales and positions as candidates to a DCNN. Factoring global and local deformations allows a DCNN to 'focus its resources' on the treatment of non-rigid deformations and yields a substantial classification accuracy improvement. Third, further pursuing this idea, we develop an efficient DCNN sliding window object detector that employs explicit search over position, scale, and aspect ratio. We provide competitive image classification and localization results on the ImageNet dataset and object detection results on the Pascal VOC 2007 benchmark.
* BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation

> The second type, share-net, applies multi-scale input images to a shared network. (Attention to Scale, 2015)

* Recurrent Convolutional Neural Networks for Scene Labeling

### 5.9. Multi-Scale Architectures (2016, PSPNet) (5)

* Fully Convolutional Networks for Semantic Segmentation
* Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
* [[Attention to Scale](https://arxiv.org/abs/1511.03339)]
    [[pdf](https://arxiv.org/pdf/1511.03339.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.03339/)]
    * Title: Attention to Scale: Scale-aware Semantic Image Segmentation
    * Year: 10 Nov `2015`
    * Authors: Liang-Chieh Chen, Yi Yang, Jiang Wang, Wei Xu, Alan L. Yuille
    * Institutions: [Baidu USA]
    * Abstract: Incorporating multi-scale features in fully convolutional neural networks (FCNs) has been a key element to achieving state-of-the-art performance on semantic image segmentation. One common way to extract multi-scale features is to feed multiple resized input images to a shared deep network and then merge the resulting features for pixelwise classification. In this work, we propose an attention mechanism that learns to softly weight the multi-scale features at each pixel location. We adapt a state-of-the-art semantic image segmentation model, which we jointly train with multi-scale input images and the attention model. The proposed attention model not only outperforms average- and max-pooling, but allows us to diagnostically visualize the importance of features at different positions and scales. Moreover, we show that adding extra supervision to the output at each scale is essential to achieving excellent performance when merging multi-scale features. We demonstrate the effectiveness of our model with extensive experiments on three challenging datasets, including PASCAL-Person-Part, PASCAL VOC 2012 and a subset of MS-COCO 2014.
    * Comments:
        * > Recent work has studied two approaches to dealing with the conflicting demands of multi-scale reasoning and full-resolution dense prediction. ... Another approach involves providing multiple rescaled versions of the image as input to the network and combining the predictions obtained for these multiple inputs (Farabet et al., 2013; Lin et al., 2015; Chen et al., 2015b). (Dilated Convolutions, 2015)
* [Hierarchical Auto-Zoom Net (HAZN)](https://arxiv.org/abs/1511.06881)
    [[pdf](https://arxiv.org/pdf/1511.06881.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06881/)]
    * Title: Zoom Better to See Clearer: Human and Object Parsing with Hierarchical Auto-Zoom Net
    * Year: 21 Nov `2015`
    * Authors: Fangting Xia, Peng Wang, Liang-Chieh Chen, Alan L. Yuille
    * Institutions: [University of California, Los Angeles]
    * Abstract: Parsing articulated objects, e.g. humans and animals, into semantic parts (e.g. body, head and arms, etc.) from natural images is a challenging and fundamental problem for computer vision. A big difficulty is the large variability of scale and location for objects and their corresponding parts. Even limited mistakes in estimating scale and location will degrade the parsing output and cause errors in boundary details. To tackle these difficulties, we propose a "Hierarchical Auto-Zoom Net" (HAZN) for object part parsing which adapts to the local scales of objects and parts. HAZN is a sequence of two "Auto-Zoom Net" (AZNs), each employing fully convolutional networks that perform two tasks: (1) predict the locations and scales of object instances (the first AZN) or their parts (the second AZN); (2) estimate the part scores for predicted object instance or part regions. Our model can adaptively "zoom" (resize) predicted image regions into their proper scales to refine the parsing. We conduct extensive experiments over the PASCAL part datasets on humans, horses, and cows. For humans, our approach significantly outperforms the state-of-the-arts by 5% mIOU and is especially better at segmenting small instances and small parts. We obtain similar improvements for parsing cows and horses over alternative methods. In summary, our strategy of first zooming into objects and then zooming into parts is very effective. It also enables us to process different regions of the image at different scales adaptively so that, for example, we do not need to waste computational resources scaling the entire image.
* Hypercolumns for Object Segmentation and Fine-grained Localization

### 5.10. Efficiency

* [Speeding up Semantic Segmentation for Autonomous Driving](https://openreview.net/forum?id=S1uHiFyyg)
    [[pdf](https://openreview.net/pdf?id=S1uHiFyyg)]
    * Title: Speeding up Semantic Segmentation for Autonomous Driving
    * Year: 15 Oct `2016`
    * Authors: Michael Treml, José Arjona-Medina, Thomas Unterthiner, Rupesh Durgesh, Felix Friedmann, Peter Schuberth, Andreas Mayr, Martin Heusel, Markus Hofmarcher, Michael Widrich, Bernhard Nessler, Sepp Hochreiter
    * Abstract: Deep learning has considerably improved semantic image segmentation. However, its high accuracy is traded against larger computational costs which makes it unsuit- able for embedded devices in self-driving cars. We propose a novel deep network architecture for image segmentation that keeps the high accuracy while being efficient enough for embedded devices. The architecture consists of ELU activation functions, a SqueezeNet-like encoder, followed by parallel dilated convolutions, and a decoder with SharpMask-like refinement modules. On the Cityscapes dataset, the new network achieves higher segmentation accuracy than other networks that are tailored to embedded devices. Simultaneously the frame-rate is still sufficiently high for the deployment in autonomous vehicles.

## 6. Graphical-Model-Based Approaches

### 6.1. Unknown

* [[Conditional random fields: Probabilistic models for segmenting and labeling sequence data](https://dl.acm.org/doi/10.5555/645530.655813)]
    [[pdf](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)]
    * Title: Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data
    * Year: 28 June `2001`
    * Authors: John D. Lafferty, Andrew McCallum, Fernando C. N. Pereira
    * Institutions: [Carnegie Mellon University], [WhizBang! Labs], [University of Pennsylvania]
    * Abstract: We present conditional random fields, a framework for building probabilistic models to segment and label sequence data. Conditional random fields offer several advantages over hidden Markov models and stochastic grammars for such tasks, including the ability to relax strong independence assumptions made in those models. Conditional random fields also avoid a fundamental limitation of maximum entropy Markov models (MEMMs) and other discriminative Markov models based on directed graphical models, which can be biased towards states with few successor states. We present iterative parameter estimation algorithms for conditional random fields and compare the performance of the resulting models to HMMs and MEMMs on synthetic and natural-language data.
* [[Semantic Image Segmentation via Deep Parsing Network](https://arxiv.org/abs/1509.02634)]
    [[pdf](https://arxiv.org/pdf/1509.02634.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1509.02634/)]
    * Title: Semantic Image Segmentation via Deep Parsing Network
    * Year: 09 Sep `2015`
    * Authors: Ziwei Liu, Xiaoxiao Li, Ping Luo, Chen Change Loy, Xiaoou Tang
    * Abstract: This paper addresses semantic image segmentation by incorporating rich information into Markov Random Field (MRF), including high-order relations and mixture of label contexts. Unlike previous works that optimized MRFs using iterative algorithm, we solve MRF by proposing a Convolutional Neural Network (CNN), namely Deep Parsing Network (DPN), which enables deterministic end-to-end computation in a single forward pass. Specifically, DPN extends a contemporary CNN architecture to model unary terms and additional layers are carefully devised to approximate the mean field algorithm (MF) for pairwise terms. It has several appealing properties. First, different from the recent works that combined CNN and MRF, where many iterations of MF were required for each training image during back-propagation, DPN is able to achieve high performance by approximating one iteration of MF. Second, DPN represents various types of pairwise terms, making many existing works as its special cases. Third, DPN makes MF easier to be parallelized and speeded up in Graphical Processing Unit (GPU). DPN is thoroughly evaluated on the PASCAL VOC 2012 dataset, where a single DPN model yields a new state-of-the-art segmentation accuracy.

### 6.2. (2014, DeepLabV1) Conditional Random Fields (CRF) (count=3)

* [[Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)]
    [[pdf](https://arxiv.org/pdf/1210.5644.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1210.5644/)]
    * Title: Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
    * Year: 20 Oct `2012`
    * Authors: Philipp Krähenbühl, Vladlen Koltun
    * Institutions: [Computer Science Department Stanford University]
    * Abstract: Most state-of-the-art techniques for multi-class image segmentation and labeling use conditional random fields defined over pixels or image regions. While region-level models often feature dense pairwise connectivity, pixel-level models are considerably larger and have only permitted sparse graph structures. In this paper, we consider fully connected CRF models defined on the complete set of pixels in an image. The resulting graphs have billions of edges, making traditional inference algorithms impractical. Our main contribution is a highly efficient approximate inference algorithm for fully connected CRF models in which the pairwise edge potentials are defined by a linear combination of Gaussian kernels. Our experiments demonstrate that dense connectivity at the pixel level substantially improves segmentation and labeling accuracy.
    * Comments:
        * > (2016, RefineNet) The atrous convolution based approach DeepLab-CRF in [5] directly output a middle-resolution score map then applies the dense CRF method [27] to refine boundaries by leveraging color contrast information.
* [[Combining the Best of Graphical Models and ConvNets for Semantic Segmentation](https://arxiv.org/abs/1412.4313)]
    [[pdf](https://arxiv.org/pdf/1412.4313.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.4313/)]
    * Title: Combining the Best of Graphical Models and ConvNets for Semantic Segmentation
    * Year: 14 Dec `2014`
    * Authors: Michael Cogswell, Xiao Lin, Senthil Purushwalkam, Dhruv Batra
    * Abstract: We present a two-module approach to semantic segmentation that incorporates Convolutional Networks (CNNs) and Graphical Models. Graphical models are used to generate a small (5-30) set of diverse segmentations proposals, such that this set has high recall. Since the number of required proposals is so low, we can extract fairly complex features to rank them. Our complex feature of choice is a novel CNN called SegNet, which directly outputs a (coarse) semantic segmentation. Importantly, SegNet is specifically trained to optimize the corpus-level PASCAL IOU loss function. To the best of our knowledge, this is the first CNN specifically designed for semantic segmentation. This two-module approach achieves $52.5\%$ on the PASCAL 2012 segmentation challenge.
    * Comments:
        * > Cogswell et al. (2014) use CRFs as a proposal mechanism for a DCNN-based reranking system. (DeepLabv1, 2014)
* [[CRF-RNN](https://arxiv.org/abs/1502.03240)]
    [[pdf](https://arxiv.org/pdf/1502.03240.pdf)]
    [vanity]
    * Title: Conditional Random Fields as Recurrent Neural Networks
    * Year: 11 Feb `2015`
    * Authors: Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, Philip H. S. Torr
    * Abstract: Pixel-level labelling tasks, such as semantic segmentation, play a central role in image understanding. Recent approaches have attempted to harness the capabilities of deep learning techniques for image recognition to tackle pixel-level labelling tasks. One central issue in this methodology is the limited capacity of deep learning techniques to delineate visual objects. To solve this problem, we introduce a new form of convolutional neural network that combines the strengths of Convolutional Neural Networks (CNNs) and Conditional Random Fields (CRFs)-based probabilistic graphical modelling. To this end, we formulate mean-field approximate inference for the Conditional Random Fields with Gaussian pairwise potentials as Recurrent Neural Networks. This network, called CRF-RNN, is then plugged in as a part of a CNN to obtain a deep network that has desirable properties of both CNNs and CRFs. Importantly, our system fully integrates CRF modelling with CNNs, making it possible to train the whole deep network end-to-end with the usual back-propagation algorithm, avoiding offline post-processing methods for object delineation. We apply the proposed method to the problem of semantic image segmentation, obtaining top results on the challenging Pascal VOC 2012 segmentation benchmark.
    * Comments:
        * > The predictive performance of FCN has been improved further by appending the FCN with a recurrent neural network (RNN) [10] and fine-tuning them on large datasets [21],[42]. The RNN layers mimic the sharp boundary delineation capabilities of CRFs while exploiting the feature representation power of FCN's. (2015, SegNet)
        * > (2015, ParseNet) Zheng et al. (2015) convert CRF learning to recurrent neural network (RNN) and use message passing to do the learning and inference.

### 6.3. MAP and CRF (Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, 2012) (count=5+5)

* [[Multiscale conditional random fields for image labeling](https://ieeexplore.ieee.org/document/1315232)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1315232)]
    * Title: Multiscale conditional random fields for image labeling
    * Year: 19 July `2004`
    * Authors: Xuming He; R.S. Zemel; M.A. Carreira-Perpinan
    * Abstract: We propose an approach to include contextual features for labeling images, in which each pixel is assigned to one of a finite set of labels. The features are incorporated into a probabilistic framework, which combines the outputs of several components. Components differ in the information they encode. Some focus on the image-label mapping, while others focus solely on patterns within the label field. Components also differ in their scale, as some focus on fine-resolution patterns while others on coarser, more global structure. A supervised version of the contrastive divergence algorithm is applied to learn these features from labeled image data. We demonstrate performance on two real-world image databases and compare it to a classifier and a Markov random field.
* [[A hierarchical field framework for unified context-based classification](https://ieeexplore.ieee.org/document/1544868)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1544868)]
    * Title: A hierarchical field framework for unified context-based classification
    * Year: 05 December `2005`
    * Authors: S. Kumar; M. Hebert
    * Abstract: We present a two-layer hierarchical formulation to exploit different levels of contextual information in images for robust classification. Each layer is modeled as a conditional field that allows one to capture arbitrary observation-dependent label interactions. The proposed framework has two main advantages. First, it encodes both the short-range interactions (e.g., pixelwise label smoothing) as well as the long-range interactions (e.g., relative configurations of objects or regions) in a tractable manner. Second, the formulation is general enough to be applied to different domains ranging from pixelwise image labeling to contextual object detection. The parameters of the model are learned using a sequential maximum-likelihood approximation. The benefits of the proposed framework are demonstrated on four different datasets and comparison results are presented
* [[Objects in Context](https://ieeexplore.ieee.org/document/4408986)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4408986)]
    * Title: Objects in Context
    * Year: 26 December `2007`
    * Authors: Andrew Rabinovich; Andrea Vedaldi; Carolina Galleguillos; Eric Wiewiora; Serge Belongie
    * Abstract: In the task of visual object categorization, semantic context can play the very important role of reducing ambiguity in objects' visual appearance. In this work we propose to incorporate semantic object context as a post-processing step into any off-the-shelf object categorization model. Using a conditional random field (CRF) framework, our approach maximizes object label agreement according to contextual relevance. We compare two sources of context: one learned from training data and another queried from Google Sets. The overall performance of the proposed framework is evaluated on the PASCAL and MSRC datasets. Our findings conclude that incorporating context into object categorization greatly improves categorization accuracy.
* [[TextonBoost for Image Understanding: Multi-Class Object Recognition and Segmentation by Jointly Modeling Texture, Layout, and Context](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.126.149)]
    [[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.126.149&rep=rep1&type=pdf)]
    * Title: TextonBoost for Image Understanding: Multi-Class Object Recognition and Segmentation by Jointly Modeling Texture, Layout, and Context
    * Year: `2007`
    * Authors: Jamie Shotton , John Winn , Carsten Rother , Antonio Criminisi
    * Institutions: [Machine Intelligence Laboratory, University of Cambridge], [Microsoft Research Cambridge, UK]
    * Abstract: This paper details a new approach for learning a discriminative model of object classes, incorporating texture, layout, and context information efficiently. The learned model is used for automatic visual understanding and semantic segmentation of photographs. Our discriminative model exploits texture-layout filters, novel features based on textons, which jointly model patterns of texture and their spatial layout. Unary classification and feature selection is achieved using shared boosting to give an efficient classifier which can be applied to a large number of classes. Accurate image segmentation is achieved by incorporating the unary classifier in a conditional random field, which (i) captures the spatial interactions between class labels of neighboring pixels, and (ii) improves the segmentation of specific object instances. Efficient training of the model on large datasets is achieved by exploiting both random feature selection and piecewise training methods. High classification and segmentation accuracy is demonstrated on four varied databases: (i) the MSRC 21-class database containing photographs of real objects viewed under general lighting conditions, poses and viewpoints, (ii) the 7-class Corel subset and (iii) the 7-class Sowerby database used in [19], and (iv) a set of video sequences of television shows. The proposed algorithm gives competitive and visually pleasing results for objects that are highly textured (grass, trees, etc.), highly structured (cars, faces, bicycles, airplanes, etc.), and even articulated (body, cow, etc.).
* [[Robust higher order potentials for enforcing label consistency](https://ieeexplore.ieee.org/document/4587417)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4587417)]
    * Title: Robust higher order potentials for enforcing label consistency
    * Year: 05 August `2008`
    * Authors: Pushmeet Kohli; L'ubor Ladicky; Philip H. S. Torr
    * Institutions: [Microsoft Research Cambridge], [Oxford Brookes University]
    * Abstract: This paper proposes a novel framework for labelling problems which is able to combine multiple segmentations in a principled manner. Our method is based on higher order conditional random fields and uses potentials defined on sets of pixels (image segments) generated using unsupervised segmentation algorithms. These potentials enforce label consistency in image regions and can be seen as a strict generalization of the commonly used pairwise contrast sensitive smoothness potentials. The higher order potential functions used in our framework take the form of the Robust P n model. This enables the use of powerful graph cut based move making algorithms for performing inference in the framework [14]. We test our method on the problem of multi-class object segmentation by augmenting the conventional CRF used for object segmentation with higher order potentials defined on image regions. Experiments on challenging data sets show that integration of higher order potentials quantitatively and qualitatively improves results leading to much better definition of object boundaries. We believe that this method can be used to yield similar improvements for many other labelling problems.

* [[Scene Segmentation with CRFs Learned from Partially Labeled Images](https://papers.nips.cc/paper/2007/hash/cb70ab375662576bd1ac5aaf16b3fca4-Abstract.html)]
    [[pdf](https://papers.nips.cc/paper/2007/file/cb70ab375662576bd1ac5aaf16b3fca4-Paper.pdf)]
    * Title: Scene Segmentation with CRFs Learned from Partially Labeled Images
    * Year: `2007`
    * Authors: Bill Triggs, Jakob Verbeek
    * Abstract: Conditional Random Fields (CRFs) are an effective tool for a variety of different data segmentation and labeling tasks including visual scene interpretation, which seeks to partition images into their constituent semantic-level regions and assign appropriate class labels to each region. For accurate labeling it is important to capture the global context of the image as well as local information. We in- troduce a CRF based scene labeling model that incorporates both local features and features aggregated over the whole image or large sections of it. Secondly, traditional CRF learning requires fully labeled datasets which can be costly and troublesome to produce. We introduce a method for learning CRFs from datasets with many unlabeled nodes by marginalizing out the unknown labels so that the log-likelihood of the known ones can be maximized by gradient ascent. Loopy Belief Propagation is used to approximate the marginals needed for the gradi- ent and log-likelihood calculations and the Bethe free-energy approximation to the log-likelihood is monitored to control the step size. Our experimental results show that effective models can be learned from fragmentary labelings and that incorporating top-down aggregate features signiﬁcantly improves the segmenta- tions. The resulting segmentations are compared to the state-of-the-art on three different image datasets.
* Multi-Class Segmentation with Relative Location Prior
* Class segmentation and object localization with superpixel neighborhoods
* [[Associative hierarchical CRFs for object class image segmentation](https://ieeexplore.ieee.org/document/5459248)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5459248)]
    * Title: Associative hierarchical CRFs for object class image segmentation
    * Year: 06 May `2010`
    * Authors: L'ubor Ladický; Chris Russell; Pushmeet Kohli; Philip H.S. Torr
    * Abstract: Most methods for object class segmentation are formulated as a labelling problem over a single choice of quantisation of an image space - pixels, segments or group of segments. It is well known that each quantisation has its fair share of pros and cons; and the existence of a common optimal quantisation level suitable for all object categories is highly unlikely. Motivated by this observation, we propose a hierarchical random field model, that allows integration of features computed at different levels of the quantisation hierarchy. MAP inference in this model can be performed efficiently using powerful graph cut based move making algorithms. Our framework generalises much of the previous work based on pixels or segments. We evaluate its efficiency on some of the most challenging data-sets for object class segmentation, and show it obtains state-of-the-art results.
* [Graph Cut based Inference with Co-occurrence Statistics](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.168.5165&rank=1&q=Graph%20Cut%20based%20Inference%20with%20Co-occurrence%20Statistics&osm=&ossid=)
    [[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.168.5165&rep=rep1&type=pdf)]
    * Title: Graph Cut based Inference with Co-occurrence Statistics
    * Year: `2010`
    * Authors: Lubor Ladicky , Chris Russell , Pushmeet Kohli , Philip H. S. Torr
    * Institutions: [Oxford Brookes], [Microsoft Research]
    * Abstract: Markov and Conditional random fields (CRFs) used in computer vision typically model only local interactions between variables, as this is computationally tractable. In this paper we consider a class of global potentials defined over all variables in the CRF. We show how they can be readily optimised using standard graph cut algorithms at little extra expense compared to a standard pairwise field. This result can be directly used for the problem of class based image segmentation which has seen increasing recent interest within computer vision. Here the aim is to assign a label to each pixel of a given image from a set of possible object classes. Typically these methods use random fields to model local interactions between pixels or super-pixels. One of the cues that helps recognition is global object co-occurrence statistics, a measure of which classes (such as chair or motorbike) are likely to occur in the same image together. There have been several approaches proposed to exploit this property, but all of them suffer from different limitations and typically carry a high computational cost, preventing their application on large images. We find that the new model we propose produces an improvement in the labelling compared to just using a pairwise model.

### 6.4. (2015, SegNet) MAP and CRF (count=2)

* [[Fully Connected Deep Structured Networks](https://arxiv.org/abs/1503.02351)]
    [[pdf](https://arxiv.org/pdf/1503.02351.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1503.02351/)]
    * Title: Fully Connected Deep Structured Networks
    * Year: 09 Mar `2015`
    * Authors: Alexander G. Schwing, Raquel Urtasun
    * Abstract: Convolutional neural networks with many layers have recently been shown to achieve excellent results on many high-level tasks such as image classification, object detection and more recently also semantic segmentation. Particularly for semantic segmentation, a two-stage procedure is often employed. Hereby, convolutional networks are trained to provide good local pixel-wise features for the second step being traditionally a more global graphical model. In this work we unify this two-stage process into a single joint training algorithm. We demonstrate our method on the semantic image segmentation task and show encouraging results on the challenging PASCAL VOC 2012 dataset.
    * Comments:
        * > (2015, ParseNet) Schwing & Urtasun (2015) propagates the marginals computed from the structured loss to update the network parameters.
* [[Efficient piecewise training of deep structured models for semantic segmentation](https://arxiv.org/abs/1504.01013)]
    [[pdf](https://arxiv.org/pdf/1504.01013.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1504.01013/)]
    * Title: Efficient piecewise training of deep structured models for semantic segmentation
    * Year: 04 Apr `2015`
    * Authors: Guosheng Lin, Chunhua Shen, Anton van dan Hengel, Ian Reid
    * Abstract: Recent advances in semantic image segmentation have mostly been achieved by training deep convolutional neural networks (CNNs). We show how to improve semantic segmentation through the use of contextual information; specifically, we explore 'patch-patch' context between image regions, and 'patch-background' context. For learning from the patch-patch context, we formulate Conditional Random Fields (CRFs) with CNN-based pairwise potential functions to capture semantic correlations between neighboring patches. Efficient piecewise training of the proposed deep structured model is then applied to avoid repeated expensive CRF inference for back propagation. For capturing the patch-background context, we show that a network design with traditional multi-scale image input and sliding pyramid pooling is effective for improving performance. Our experimental results set new state-of-the-art performance on a number of popular semantic segmentation datasets, including NYUDv2, PASCAL VOC 2012, PASCAL-Context, and SIFT-flow. In particular, we achieve an intersection-over-union score of 78.0 on the challenging PASCAL VOC 2012 dataset.
    * Comments:
        * > Lin et al. [34] resized the input image for three scales and concatenated the resulting three-scale features to generate the unary and pairwise potentials of a Conditional Random Field (CRF). (Attention to Scale, 2015)
        * > (2015, ParseNet) Lin et al. (2015) uses piece-wise training to make learning more efficient by adding a few extra piece-wise networks.

### 6.5. Fully Connected CRFs (Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, 2012) (4)

* Objects in Context
* [[Random Field Model for Integration of Local Information and Global Information](https://ieeexplore.ieee.org/document/4497207)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4497207)]
    * Title: Random Field Model for Integration of Local Information and Global Information
    * Year: 20 June `2008`
    * Authors: Takahiro Toyoda; Osamu Hasegawa
    * Abstract: This paper presents a proposal of a general framework that explicitly models local information and global information in a conditional random field. The proposed method extracts global image features as well as local ones and uses them to predict the scene of the input image. Scene-based top-down information is generated based on the predicted scene. It represents a global spatial configuration of labels and category compatibility over an image. Incorporation of the global information helps to resolve local ambiguities and achieves locally and globally consistent image recognition. In spite of the model's simplicity, the proposed method demonstrates good performance in image labeling of two datasets.
* [[Object categorization using co-occurrence, location and appearance](https://ieeexplore.ieee.org/document/4587799)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4587799)]
    * Title: Object categorization using co-occurrence, location and appearance
    * Year: 05 August `2008`
    * Authors: Carolina Galleguillos; Andrew Rabinovich; Serge Belongie
    * Abstract: In this work we introduce a novel approach to object categorization that incorporates two types of context-co-occurrence and relative location - with local appearance-based features. Our approach, named CoLA (for co-occurrence, location and appearance), uses a conditional random field (CRF) to maximize object label agreement according to both semantic and spatial relevance. We model relative location between objects using simple pairwise features. By vector quantizing this feature space, we learn a small set of prototypical spatial relationships directly from the data. We evaluate our results on two challenging datasets: PASCAL 2007 and MSRC. The results show that combining co-occurrence and spatial context improves accuracy in as many as half of the categories compared to using co-occurrence alone.
* [[(RF)^2 -- Random Forest Random Field](https://papers.nips.cc/paper/2010/hash/289dff07669d7a23de0ef88d2f7129e7-Abstract.html)]
    [[pdf](https://papers.nips.cc/paper/2010/file/289dff07669d7a23de0ef88d2f7129e7-Paper.pdf)]
    * Title: (RF)^2 -- Random Forest Random Field
    * Year: `2010`
    * Authors: Nadia Payet, Sinisa Todorovic
    * Abstract: We combine random forest (RF) and conditional random field (CRF) into a new computational framework, called random forest random field (RF)^2. Inference of (RF)^2 uses the Swendsen-Wang cut algorithm, characterized by Metropolis-Hastings jumps. A jump from one state to another depends on the ratio of the proposal distributions, and on the ratio of the posterior distributions of the two states. Prior work typically resorts to a parametric estimation of these four distributions, and then computes their ratio. Our key idea is to instead directly estimate these ratios using RF. RF collects in leaf nodes of each decision tree the class histograms of training examples. We use these class histograms for a non-parametric estimation of the distribution ratios. We derive the theoretical error bounds of a two-class (RF)^2. (RF)^2 is applied to a challenging task of multiclass object recognition and segmentation over a random field of input image regions. In our empirical evaluation, we use only the visual information provided by image regions (e.g., color, texture, spatial layout), whereas the competing methods additionally use higher-level cues about the horizon location and 3D layout of surfaces in the scene. Nevertheless, (RF)^2 outperforms the state of the art on benchmark datasets, in terms of accuracy and computation time.

### 6.6. DeepLab Family

* [[DeepLabV1](https://arxiv.org/abs/1412.7062)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1412.7062.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1412.7062/)]
    * Title: Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
    * Year: 22 Dec `2014`
    * Authors: Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
    * Institutions: [Univ. of California, Los Angeles], [Google Inc.], [CentraleSupélec and INRIA]
    * Abstract: Deep Convolutional Neural Networks (DCNNs) have recently shown state of the art performance in high level vision tasks, such as image classification and object detection. This work brings together methods from DCNNs and probabilistic graphical models for addressing the task of pixel-level classification (also called "semantic image segmentation"). We show that responses at the final layer of DCNNs are not sufficiently localized for accurate object segmentation. This is due to the very invariance properties that make DCNNs good for high level tasks. We overcome this poor localization property of deep networks by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF). Qualitatively, our "DeepLab" system is able to localize segment boundaries at a level of accuracy which is beyond previous methods. Quantitatively, our method sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 71.6% IOU accuracy in the test set. We show how these results can be obtained efficiently: Careful network re-purposing and a novel application of the 'hole' algorithm from the wavelet community allow dense computation of neural net responses at 8 frames per second on a modern GPU.
    * Comments:
        * DeepLab is a variant of FCNs.
        * > (2015, Dilated Convolutions) Chen et al. (2015a) used dilation to simplify the architecture of Long et al. (2015).
        * > (2015, SegNet) The method of [3] also use the feature maps of the classification network with an independent CRF post-processing technique to perform segmentation.
        * > (2015, ParseNet) Chen et al. (2014) first uses a FCN to estimate the unary potential, then applies a fully connected CRF to smooth the predictions spatially.
        * > (2016, ENet) Other existing architectures use simpler classifiers and then cascade them with Conditional Random Field (CRF) as a post-processing step liang14 ; sturgess09 .
        * > (2016, RefineNet) The atrous convolution based approach DeepLab-CRF in [5] directly output a middle-resolution score map then applies the dense CRF method [27] to refine boundaries by leveraging color contrast information.
* [[DeepLabV2](https://arxiv.org/abs/1606.00915)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1606.00915.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1606.00915/)]
    * Title: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
    * Year: 02 Jun `2016`
    * Authors: Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
    * Abstract: In this work we address the task of semantic image segmentation with Deep Learning and make three main contributions that are experimentally shown to have substantial practical merit. First, we highlight convolution with upsampled filters, or 'atrous convolution', as a powerful tool in dense prediction tasks. Atrous convolution allows us to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. It also allows us to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation. Second, we propose atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales. Third, we improve the localization of object boundaries by combining methods from DCNNs and probabilistic graphical models. The commonly deployed combination of max-pooling and downsampling in DCNNs achieves invariance but has a toll on localization accuracy. We overcome this by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance. Our proposed "DeepLab" system sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 79.7% mIOU in the test set, and advances the results on three other datasets: PASCAL-Context, PASCAL-Person-Part, and Cityscapes. All of our code is made publicly available online.
    * Comments:
        * > (2016, RefineNet) The method DeepLab recently proposed by Chen et al. [6] employs atrous (or dilated) convolutions to account for larger receptive fields without downscaling the image.
        * > (2017, ERFNet) The work in [8] (DeepLab2) combines a ResNet-101 with spatial pyramid pooling and CRF to reach state-of-the-art segmentation accuracy.
        * > (2018, ESPNetV1) An ASP module [3], shown in Fig. 3e, is built on the principle of split-transform-merge. The ASP module involves branching with each branch learning kernel at a different receptive field (using dilated convolutions).
* [[DeepLabV3](https://arxiv.org/abs/1706.05587)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1706.05587.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1706.05587/)]
    * Title: Rethinking Atrous Convolution for Semantic Image Segmentation
    * Year: 17 Jun `2017`
    * Authors: Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
    * Abstract: In this work, we revisit atrous convolution, a powerful tool to explicitly adjust filter's field-of-view as well as control the resolution of feature responses computed by Deep Convolutional Neural Networks, in the application of semantic image segmentation. To handle the problem of segmenting objects at multiple scales, we design modules which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates. Furthermore, we propose to augment our previously proposed Atrous Spatial Pyramid Pooling module, which probes convolutional features at multiple scales, with image-level features encoding global context and further boost performance. We also elaborate on implementation details and share our experience on training our system. The proposed `DeepLabv3' system significantly improves over our previous DeepLab versions without DenseCRF post-processing and attains comparable performance with other state-of-art models on the PASCAL VOC 2012 semantic image segmentation benchmark.
* [[DeepLabV3+](https://arxiv.org/abs/1802.02611)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1802.02611.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1802.02611/)]
    * Title: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
    * Year: 07 Feb `2018`
    * Authors: Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
    * Abstract: Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information. In this work, we propose to combine the advantages from both methods. Specifically, our proposed model, DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. We further explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network. We demonstrate the effectiveness of the proposed model on PASCAL VOC 2012 and Cityscapes datasets, achieving the test set performance of 89.0\% and 82.1\% without any post-processing. Our paper is accompanied with a publicly available reference implementation of the proposed models in Tensorflow at \url{this https URL}.
    * Comments:
        * > To increase feature resolution, which is necessary for generating high-quality results, recent top methods [12, 56, 5, 57] rely heavily on the use of dilated convolution [55] (also known as atrous convolution [10]). While effective, such an approach can substantially increase compute and memory, limiting the type of backbone network that can be used. (Panoptic FPN, 2019)
* [Auto-DeepLab](https://arxiv.org/abs/1901.02985)
    [[pdf](https://arxiv.org/pdf/1901.02985.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1901.02985/)]
    * Title: Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation
    * Year: 10 Jan `2019`
    * Authors: Chenxi Liu, Liang-Chieh Chen, Florian Schroff, Hartwig Adam, Wei Hua, Alan Yuille, Li Fei-Fei
    * Abstract: Recently, Neural Architecture Search (NAS) has successfully identified neural network architectures that exceed human designed ones on large-scale image classification. In this paper, we study NAS for semantic image segmentation. Existing works often focus on searching the repeatable cell structure, while hand-designing the outer network structure that controls the spatial resolution changes. This choice simplifies the search space, but becomes increasingly problematic for dense image prediction which exhibits a lot more network level architectural variations. Therefore, we propose to search the network level structure in addition to the cell level structure, which forms a hierarchical architecture search space. We present a network level search space that includes many popular designs, and develop a formulation that allows efficient gradient-based architecture search (3 P100 GPU days on Cityscapes images). We demonstrate the effectiveness of the proposed method on the challenging Cityscapes, PASCAL VOC 2012, and ADE20K datasets. Auto-DeepLab, our architecture searched specifically for semantic image segmentation, attains state-of-the-art performance without any ImageNet pretraining.

## 7. Attention Mechanism

### 7.1. (2022, SegNeXt) (count=6)

* [[DANet](https://ieeexplore.ieee.org/document/8953974)]
    * Title: Dual Attention Network for Scene Segmentation
    * Year: 09 January `2020`
    * Author: Jun Fu
    * Abstract: In this paper, we address the scene segmentation task by capturing rich contextual dependencies based on the self-attention mechanism. Unlike previous works that capture contexts by multi-scale features fusion, we propose a Dual Attention Networks (DANet) to adaptively integrate local features with their global dependencies. Specifically, we append two types of attention modules on top of traditional dilated FCN, which model the semantic interdependencies in spatial and channel dimensions respectively. The position attention module selectively aggregates the features at each position by a weighted sum of the features at all positions. Similar features would be related to each other regardless of their distances. Meanwhile, the channel attention module selectively emphasizes interdependent channel maps by integrating associated features among all channel maps. We sum the outputs of the two attention modules to further improve feature representation which contributes to more precise segmentation results. We achieve new state-of-the-art segmentation performance on three challenging scene segmentation datasets, i.e., Cityscapes, PASCAL Context and COCO Stuff dataset. In particular, a Mean IoU score of 81.5% on Cityscapes test set is achieved without using coarse data.
* [[CCNet](https://arxiv.org/abs/1811.11721)]
    [[pdf](https://arxiv.org/pdf/1811.11721.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1811.11721/)]
    * Title: CCNet: Criss-Cross Attention for Semantic Segmentation
    * Year: 28 Nov `2018`
    * Authors: Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, Thomas S. Huang
    * Abstract: Contextual information is vital in visual understanding problems, such as semantic segmentation and object detection. We propose a Criss-Cross Network (CCNet) for obtaining full-image contextual information in a very effective and efficient way. Concretely, for each pixel, a novel criss-cross attention module harvests the contextual information of all the pixels on its criss-cross path. By taking a further recurrent operation, each pixel can finally capture the full-image dependencies. Besides, a category consistent loss is proposed to enforce the criss-cross attention module to produce more discriminative features. Overall, CCNet is with the following merits: 1) GPU memory friendly. Compared with the non-local block, the proposed recurrent criss-cross attention module requires 11x less GPU memory usage. 2) High computational efficiency. The recurrent criss-cross attention significantly reduces FLOPs by about 85% of the non-local block. 3) The state-of-the-art performance. We conduct extensive experiments on semantic segmentation benchmarks including Cityscapes, ADE20K, human parsing benchmark LIP, instance segmentation benchmark COCO, video segmentation benchmark CamVid. In particular, our CCNet achieves the mIoU scores of 81.9%, 45.76% and 55.47% on the Cityscapes test set, the ADE20K validation set and the LIP validation set respectively, which are the new state-of-the-art results. The source codes are available at \url{this https URL}.
* [[OCNet](https://arxiv.org/abs/1809.00916)]
    [[pdf](https://arxiv.org/pdf/1809.00916.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1809.00916/)]
    * Title: OCNet: Object Context Network for Scene Parsing
    * Year: 04 Sep `2018`
    * Authors: Yuhui Yuan, Lang Huang, Jianyuan Guo, Chao Zhang, Xilin Chen, Jingdong Wang
    * Abstract: In this paper, we address the semantic segmentation task with a new context aggregation scheme named \emph{object context}, which focuses on enhancing the role of object information. Motivated by the fact that the category of each pixel is inherited from the object it belongs to, we define the object context for each pixel as the set of pixels that belong to the same category as the given pixel in the image. We use a binary relation matrix to represent the relationship between all pixels, where the value one indicates the two selected pixels belong to the same category and zero otherwise. We propose to use a dense relation matrix to serve as a surrogate for the binary relation matrix. The dense relation matrix is capable to emphasize the contribution of object information as the relation scores tend to be larger on the object pixels than the other pixels. Considering that the dense relation matrix estimation requires quadratic computation overhead and memory consumption w.r.t. the input size, we propose an efficient interlaced sparse self-attention scheme to model the dense relations between any two of all pixels via the combination of two sparse relation matrices. To capture richer context information, we further combine our interlaced sparse self-attention scheme with the conventional multi-scale context schemes including pyramid pooling~\citep{zhao2017pyramid} and atrous spatial pyramid pooling~\citep{chen2018deeplab}. We empirically show the advantages of our approach with competitive performances on five challenging benchmarks including: Cityscapes, ADE20K, LIP, PASCAL-Context and COCO-Stuff
* [[EMANet](https://arxiv.org/abs/1907.13426)]
    [[pdf](https://arxiv.org/pdf/1907.13426.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1907.13426/)]
    * Title: Expectation-Maximization Attention Networks for Semantic Segmentation
    * Year: 31 Jul `2019`
    * Authors: Xia Li, Zhisheng Zhong, Jianlong Wu, Yibo Yang, Zhouchen Lin, Hong Liu
    * Abstract: Self-attention mechanism has been widely used for various tasks. It is designed to compute the representation of each position by a weighted sum of the features at all positions. Thus, it can capture long-range relations for computer vision tasks. However, it is computationally consuming. Since the attention maps are computed w.r.t all other positions. In this paper, we formulate the attention mechanism into an expectation-maximization manner and iteratively estimate a much more compact set of bases upon which the attention maps are computed. By a weighted summation upon these bases, the resulting representation is low-rank and deprecates noisy information from the input. The proposed Expectation-Maximization Attention (EMA) module is robust to the variance of input and is also friendly in memory and computation. Moreover, we set up the bases maintenance and normalization methods to stabilize its training procedure. We conduct extensive experiments on popular semantic segmentation benchmarks including PASCAL VOC, PASCAL Context and COCO Stuff, on which we set new records.
* [[APCNet](https://ieeexplore.ieee.org/document/8954288)]
    * Title: Adaptive Pyramid Context Network for Semantic Segmentation
    * Year: 09 January `2020`
    * Authors: Junjun He; Zhongying Deng; Lei Zhou; Yali Wang; Yu Qiao
    * Abstract: Recent studies witnessed that context features can significantly improve the performance of deep semantic segmentation networks. Current context based segmentation methods differ with each other in how to construct context features and perform differently in practice. This paper firstly introduces three desirable properties of context features in segmentation task. Specially, we find that Global-guided Local Affinity (GLA) can play a vital role in constructing effective context features, while this property has been largely ignored in previous works. Based on this analysis, this paper proposes Adaptive Pyramid Context Network (APCNet) for semantic segmentation. APCNet adaptively constructs multi-scale contextual representations with multiple well-designed Adaptive Context Modules (ACMs). Specifically, each ACM leverages a global image representation as a guidance to estimate the local affinity coefficients for each sub-region, and then calculates a context vector with these affinities. We empirically evaluate our APCNet on three semantic segmentation and scene parsing datasets, including PASCAL VOC 2012, Pascal-Context, and ADE20K dataset. Experimental results show that APCNet achieves state-of-the-art performance on all three benchmarks, and obtains a new record 84.2% on PASCAL VOC 2012 test set without MS COCO pre-trained and any post-processing.
* [[Context Encoding](https://arxiv.org/abs/1803.08904)]
    [[pdf](https://arxiv.org/pdf/1803.08904.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1803.08904/)]
    * Title: Context Encoding for Semantic Segmentation
    * Year: 23 Mar `2018`
    * Authors: Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal
    * Abstract: Recent work has made significant progress in improving spatial resolution for pixelwise labeling with Fully Convolutional Network (FCN) framework by employing Dilated/Atrous convolution, utilizing multi-scale features and refining boundaries. In this paper, we explore the impact of global contextual information in semantic segmentation by introducing the Context Encoding Module, which captures the semantic context of scenes and selectively highlights class-dependent featuremaps. The proposed Context Encoding Module significantly improves semantic segmentation results with only marginal extra computation cost over FCN. Our approach has achieved new state-of-the-art results 51.7% mIoU on PASCAL-Context, 85.9% mIoU on PASCAL VOC 2012. Our single model achieves a final score of 0.5567 on ADE20K test set, which surpass the winning entry of COCO-Place Challenge in 2017. In addition, we also explore how the Context Encoding Module can improve the feature representation of relatively shallow networks for the image classification on CIFAR-10 dataset. Our 14 layer network has achieved an error rate of 3.45%, which is comparable with state-of-the-art approaches with over 10 times more layers. The source code for the complete system are publicly available.

## 8. Transformer-Architectures-Based Approaches

### 8.1. (2022, SegNeXt) (count=8)

* [[SETR](https://arxiv.org/abs/2012.15840)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2012.15840.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2012.15840/)]
    * Title: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers
    * Year: 31 Dec `2020`
    * Authors: Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip H.S. Torr, Li Zhang
    * Abstract: Most recent semantic segmentation methods adopt a fully-convolutional network (FCN) with an encoder-decoder architecture. The encoder progressively reduces the spatial resolution and learns more abstract/semantic visual concepts with larger receptive fields. Since context modeling is critical for segmentation, the latest efforts have been focused on increasing the receptive field, through either dilated/atrous convolutions or inserting attention modules. However, the encoder-decoder based FCN architecture remains unchanged. In this paper, we aim to provide an alternative perspective by treating semantic segmentation as a sequence-to-sequence prediction task. Specifically, we deploy a pure transformer (ie, without convolution and resolution reduction) to encode an image as a sequence of patches. With the global context modeled in every layer of the transformer, this encoder can be combined with a simple decoder to provide a powerful segmentation model, termed SEgmentation TRansformer (SETR). Extensive experiments show that SETR achieves new state of the art on ADE20K (50.28% mIoU), Pascal Context (55.83% mIoU) and competitive results on Cityscapes. Particularly, we achieve the first position in the highly competitive ADE20K test server leaderboard on the day of submission.
    * Comments:
        * > (2022, Recent Advances) Zheng et al. (2021) introduced a sequence-to-sequence approach and replaced the conv-encoder with a pure transformer.
* [[SegFormer](https://arxiv.org/abs/2105.15203)]
    [[pdf](https://arxiv.org/pdf/2105.15203.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2105.15203/)]
    * Title: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    * Year: 31 May `2021`
    * Authors: Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo
    * Abstract: We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perception (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding, thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from different layers, and thus combining both local attention and global attention to render powerful representations. We show that this simple and lightweight design is the key to efficient segmentation on Transformers. We scale our approach up to obtain a series of models from SegFormer-B0 to SegFormer-B5, reaching significantly better performance and efficiency than previous counterparts. For example, SegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters, being 5x smaller and 2.2% better than the previous best method. Our best model, SegFormer-B5, achieves 84.0% mIoU on Cityscapes validation set and shows excellent zero-shot robustness on Cityscapes-C. Code will be released at: this http URL.
    * Comments:
        * > (2022, Recent Advances) Xie et al. (2021) presented SegFormer, a simple yet powerful method with lightweight MLP decoders. An encoder is based on a hierarchical structure that gives multi-scale features and does not require any positional encoding scheme. SegFormer gets rid of complex decoders combining local attention and global attention to gender representation.
* [[HRFormer](https://arxiv.org/abs/2110.09408)]
    [[pdf](https://arxiv.org/pdf/2110.09408.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2110.09408/)]
    * Title: HRFormer: High-Resolution Transformer for Dense Prediction
    * Year: 18 Oct `2021`
    * Authors: Yuhui Yuan, Rao Fu, Lang Huang, Weihong Lin, Chao Zhang, Xilin Chen, Jingdong Wang
    * Abstract: We present a High-Resolution Transformer (HRFormer) that learns high-resolution representations for dense prediction tasks, in contrast to the original Vision Transformer that produces low-resolution representations and has high memory and computational cost. We take advantage of the multi-resolution parallel design introduced in high-resolution convolutional networks (HRNet), along with local-window self-attention that performs self-attention over small non-overlapping image windows, for improving the memory and computation efficiency. In addition, we introduce a convolution into the FFN to exchange information across the disconnected image windows. We demonstrate the effectiveness of the High-Resolution Transformer on both human pose estimation and semantic segmentation tasks, e.g., HRFormer outperforms Swin transformer by $1.3$ AP on COCO pose estimation with $50\%$ fewer parameters and $30\%$ fewer FLOPs. Code is available at: this https URL.
* [[Segmenter](https://arxiv.org/abs/2105.05633)]
    [[pdf](https://arxiv.org/pdf/2105.05633.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2105.05633/)]
    * Title: Segmenter: Transformer for Semantic Segmentation
    * Year: 12 May `2021`
    * Authors: Robin Strudel, Ricardo Garcia, Ivan Laptev, Cordelia Schmid
    * Abstract: Image segmentation is often ambiguous at the level of individual image patches and requires contextual information to reach label consensus. In this paper we introduce Segmenter, a transformer model for semantic segmentation. In contrast to convolution-based methods, our approach allows to model global context already at the first layer and throughout the network. We build on the recent Vision Transformer (ViT) and extend it to semantic segmentation. To do so, we rely on the output embeddings corresponding to image patches and obtain class labels from these embeddings with a point-wise linear decoder or a mask transformer decoder. We leverage models pre-trained for image classification and show that we can fine-tune them on moderate sized datasets available for semantic segmentation. The linear decoder allows to obtain excellent results already, but the performance can be further improved by a mask transformer generating class masks. We conduct an extensive ablation study to show the impact of the different parameters, in particular the performance is better for large models and small patch sizes. Segmenter attains excellent results for semantic segmentation. It outperforms the state of the art on both ADE20K and Pascal Context datasets and is competitive on Cityscapes.
* [[DPT](https://arxiv.org/abs/2103.13413)]
    [[pdf](https://arxiv.org/pdf/2103.13413.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.13413/)]
    * Title: Vision Transformers for Dense Prediction
    * Year: 24 Mar `2021`
    * Authors: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
    * Abstract: We introduce dense vision transformers, an architecture that leverages vision transformers in place of convolutional networks as a backbone for dense prediction tasks. We assemble tokens from various stages of the vision transformer into image-like representations at various resolutions and progressively combine them into full-resolution predictions using a convolutional decoder. The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent predictions when compared to fully-convolutional networks. Our experiments show that this architecture yields substantial improvements on dense prediction tasks, especially when a large amount of training data is available. For monocular depth estimation, we observe an improvement of up to 28% in relative performance when compared to a state-of-the-art fully-convolutional network. When applied to semantic segmentation, dense vision transformers set a new state of the art on ADE20K with 49.02% mIoU. We further show that the architecture can be fine-tuned on smaller datasets such as NYUv2, KITTI, and Pascal Context where it also sets the new state of the art. Our models are available at this https URL.
* [[Video K-Net](https://arxiv.org/abs/2204.04656)]
    [[pdf](https://arxiv.org/pdf/2204.04656.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2204.04656/)]
    * Title: Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation
    * Year: 10 Apr `2022`
    * Authors: Xiangtai Li, Wenwei Zhang, Jiangmiao Pang, Kai Chen, Guangliang Cheng, Yunhai Tong, Chen Change Loy
    * Abstract: This paper presents Video K-Net, a simple, strong, and unified framework for fully end-to-end video panoptic segmentation. The method is built upon K-Net, a method that unifies image segmentation via a group of learnable kernels. We observe that these learnable kernels from K-Net, which encode object appearances and contexts, can naturally associate identical instances across video frames. Motivated by this observation, Video K-Net learns to simultaneously segment and track "things" and "stuff" in a video with simple kernel-based appearance modeling and cross-temporal kernel interaction. Despite the simplicity, it achieves state-of-the-art video panoptic segmentation results on Citscapes-VPS, KITTI-STEP, and VIPSeg without bells and whistles. In particular, on KITTI-STEP, the simple method can boost almost 12\% relative improvements over previous methods. On VIPSeg, Video K-Net boosts almost 15\% relative improvements and results in 39.8 % VPQ. We also validate its generalization on video semantic segmentation, where we boost various baselines by 2\% on the VSPW dataset. Moreover, we extend K-Net into clip-level video framework for video instance segmentation, where we obtain 40.5% mAP for ResNet50 backbone and 54.1% mAP for Swin-base on YouTube-2019 validation set. We hope this simple, yet effective method can serve as a new, flexible baseline in unified video segmentation design. Both code and models are released at this https URL.
* [[MaskFormer](https://arxiv.org/abs/2107.06278)]
    [[pdf](https://arxiv.org/pdf/2107.06278.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2107.06278/)]
    * Title: Per-Pixel Classification is Not All You Need for Semantic Segmentation
    * Year: 13 Jul `2021`
    * Authors: Bowen Cheng, Alexander G. Schwing, Alexander Kirillov
    * Abstract: Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a single global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models.
* [[Mask2Former](https://arxiv.org/abs/2112.01527)]
    [[pdf](https://arxiv.org/pdf/2112.01527.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2112.01527/)]
    * Title: Masked-attention Mask Transformer for Universal Image Segmentation
    * Year: 02 Dec `2021`
    * Authors: Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar
    * Abstract: Image segmentation is about grouping pixels with different semantics, e.g., category or instance membership, where each choice of semantics defines a task. While only the semantics of each task differ, current research focuses on designing specialized architectures for each task. We present Masked-attention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components include masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most notably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU on ADE20K).

### 8.2. Others

* [[Fully Transformer Networks for Semantic Image Segmentation](https://arxiv.org/abs/2106.04108)]
    [[pdf](https://arxiv.org/pdf/2106.04108.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.04108/)]
    * Title: Fully Transformer Networks for Semantic Image Segmentation
    * Year: 08 Jun `2021`
    * Authors: Sitong Wu, Tianyi Wu, Fangjian Lin, Shengwei Tian, Guodong Guo
    * Abstract: Transformers have shown impressive performance in various natural language processing and computer vision tasks, due to the capability of modeling long-range dependencies. Recent progress has demonstrated that combining such Transformers with CNN-based semantic image segmentation models is very promising. However, it is not well studied yet on how well a pure Transformer based approach can achieve for image segmentation. In this work, we explore a novel framework for semantic image segmentation, which is encoder-decoder based Fully Transformer Networks (FTN). Specifically, we first propose a Pyramid Group Transformer (PGT) as the encoder for progressively learning hierarchical features, meanwhile reducing the computation complexity of the standard Visual Transformer (ViT). Then, we propose a Feature Pyramid Transformer (FPT) to fuse semantic-level and spatial-level information from multiple levels of the PGT encoder for semantic image segmentation. Surprisingly, this simple baseline can achieve better results on multiple challenging semantic segmentation and face parsing benchmarks, including PASCAL Context, ADE20K, COCOStuff, and CelebAMask-HQ. The source code will be released on this https URL.
    * Comments:
        * > (2022, Recent Advances) Wu et al. (2021b) proposed a fully transformer network which relies on pyramid group transformer encoder to progress learned hierarchical features, while reducing the computational cost of standard ViT.
* [[Cross-Modal Self-Attention Network for Referring Image Segmentation](https://arxiv.org/abs/1904.04745)]
    [[pdf](https://arxiv.org/pdf/1904.04745.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.04745/)]
    * Title: Cross-Modal Self-Attention Network for Referring Image Segmentation
    * Year: 09 Apr `2019`
    * Authors: Linwei Ye, Mrigank Rochan, Zhi Liu, Yang Wang
    * Abstract: We consider the problem of referring image segmentation. Given an input image and a natural language expression, the goal is to segment the object referred by the language expression in the image. Existing works in this area treat the language expression and the input image separately in their representations. They do not sufficiently capture long-range correlations between these two modalities. In this paper, we propose a cross-modal self-attention (CMSA) module that effectively captures the long-range dependencies between linguistic and visual features. Our model can adaptively focus on informative words in the referring expression and important regions in the input image. In addition, we propose a gated multi-level fusion module to selectively integrate self-attentive cross-modal features corresponding to different levels in the image. This module controls the information flow of features at different levels. We validate the proposed approach on four evaluation datasets. Our proposed approach consistently outperforms existing state-of-the-art methods.
* [[OCRNet](https://arxiv.org/abs/1909.11065)]
    [[pdf](https://arxiv.org/pdf/1909.11065.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1909.11065/)]
    * Title: Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation
    * Year: 24 Sep `2019`
    * Authors: Yuhui Yuan, Xiaokang Chen, Xilin Chen, Jingdong Wang
    * Abstract: In this paper, we address the semantic segmentation problem with a focus on the context aggregation strategy. Our motivation is that the label of a pixel is the category of the object that the pixel belongs to. We present a simple yet effective approach, object-contextual representations, characterizing a pixel by exploiting the representation of the corresponding object class. First, we learn object regions under the supervision of ground-truth segmentation. Second, we compute the object region representation by aggregating the representations of the pixels lying in the object region. Last, % the representation similarity we compute the relation between each pixel and each object region and augment the representation of each pixel with the object-contextual representation which is a weighted aggregation of all the object region representations according to their relations with the pixel. We empirically demonstrate that the proposed approach achieves competitive performance on various challenging semantic segmentation benchmarks: Cityscapes, ADE20K, LIP, PASCAL-Context, and COCO-Stuff. Cityscapes, ADE20K, LIP, PASCAL-Context, and COCO-Stuff. Our submission "HRNet + OCR + SegFix" achieves 1-st place on the Cityscapes leaderboard by the time of submission. Code is available at: this https URL and this https URL. We rephrase the object-contextual representation scheme using the Transformer encoder-decoder framework. The details are presented in~Section3.3.

## 9. Instance Segmentation

### 9.1. Segment Proposals

* [[Simultaneous Detection and Segmentation (SDS)](https://arxiv.org/abs/1407.1808)]
    [[pdf](https://arxiv.org/pdf/1407.1808.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1407.1808/)]
    * Title: Simultaneous Detection and Segmentation
    * Year: 07 Jul `2014`
    * Authors: Bharath Hariharan, Pablo Arbeláez, Ross Girshick, Jitendra Malik
    * Institutions: [University of California, Berkeley], [Universidad de los Andes, Colombia]
    * Abstract: We aim to detect all instances of a category in an image and, for each instance, mark the pixels that belong to it. We call this task Simultaneous Detection and Segmentation (SDS). Unlike classical bounding box detection, SDS requires a segmentation and not just a box. Unlike classical semantic segmentation, we require individual object instances. We build on recent work that uses convolutional neural networks to classify category-independent region proposals (R-CNN [16]), introducing a novel architecture tailored for SDS. We then use category-specific, top- down figure-ground predictions to refine our bottom-up proposals. We show a 7 point boost (16% relative) over our baselines on SDS, a 5 point boost (10% relative) over state-of-the-art on semantic segmentation, and state-of-the-art performance in object detection. Finally, we provide diagnostic tools that unpack performance and provide directions for future work.
    * Comments:
        * > SDS differs from classical bounding box detection in its requirement of a segmentation and from classical semantic segmentation in its requirement of separate instances. (Hypercolumns, 2014)
* [[Hypercolumns](https://arxiv.org/abs/1411.5752)]
    [[pdf](https://arxiv.org/pdf/1411.5752.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1411.5752/)]
    * Title: Hypercolumns for Object Segmentation and Fine-grained Localization
    * Year: 21 Nov `2014`
    * Authors: Bharath Hariharan, Pablo Arbeláez, Ross Girshick, Jitendra Malik
    * Institutions: [University of California, Berkeley], [Universidad de los Andes, Colombia], [Microsoft Research, Redmond]
    * Abstract: Recognition algorithms based on convolutional networks (CNNs) typically use the output of the last layer as feature representation. However, the information in this layer may be too coarse to allow precise localization. On the contrary, earlier layers may be precise in localization but will not capture semantics. To get the best of both worlds, we define the hypercolumn at a pixel as the vector of activations of all CNN units above that pixel. Using hypercolumns as pixel descriptors, we show results on three fine-grained localization tasks: simultaneous detection and segmentation[22], where we improve state-of-the-art from 49.7[22] mean AP^r to 60.0, keypoint localization, where we get a 3.3 point boost over[20] and part labeling, where we show a 6.6 point gain over a strong baseline.
    * Comments:
        * > (2014, DeepLabV1) More recently, Hariharan et al. (2014a) propose to concatenate the computed inter-mediate feature maps within the DCNNs for pixel classification.
        * > (2015, U-Net) More recent approaches [11,4] proposed a classifier output that takes into account the features from multiple layers.
        * > (2015, Attention to Scale) Hariharan et al. [27] classified a pixel with hypercolumn representation (i.e., concatenation of features from intermediate layers).
        * > (2016, RefineNet) The method Hypercolumn [22] merges features from middle layers and learns dense classification layers.
* [[DeepMask](https://arxiv.org/abs/1506.06204)]
    [[pdf](https://arxiv.org/pdf/1506.06204.pdf)]
    [vanity]
    * Title: Learning to Segment Object Candidates
    * Year: 20 Jun `2015`
    * Authors: Pedro O. Pinheiro, Ronan Collobert, Piotr Dollar
    * Institutions: [Facebook AI Research]
    * Abstract: Recent object detection systems rely on two critical steps: (1) a set of object proposals is predicted as efficiently as possible, and (2) this set of candidate proposals is then passed to an object classifier. Such approaches have been shown they can be fast, while achieving the state of the art in detection performance. In this paper, we propose a new way to generate object proposals, introducing an approach based on a discriminative convolutional network. Our model is trained jointly with two objectives: given an image patch, the first part of the system outputs a class-agnostic segmentation mask, while the second part of the system outputs the likelihood of the patch being centered on a full object. At test time, the model is efficiently applied on the whole test image and generates a set of segmentation masks, each of them being assigned with a corresponding object likelihood score. We show that our model yields significant improvements over state-of-the-art object proposal algorithms. In particular, compared to previous approaches, our model obtains substantially higher object recall using fewer proposals. We also show that our model is able to generalize to unseen categories it has not seen during training. Unlike all previous approaches for generating object masks, we do not rely on edges, superpixels, or any other form of low-level segmentation.
    * Comments:
        * > DeepMask is trained to jointly generate a class-agnostic object mask and an associated 'objectness' score for each input image patch. At inference time, the model is run convolutionally to generate a dense set of scored segmentation proposals. (SharpMask, 2016)
        * > DeepMask generates masks that are accurate on the object level but only coarsely align with object boundaries. (SharpMask, 2016)
* [[SharpMask](https://arxiv.org/abs/1603.08695)]
    [[pdf](https://arxiv.org/pdf/1603.08695.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1603.08695/)]
    * Title: Learning to Refine Object Segments
    * Year: 29 Mar `2016`
    * Authors: Pedro O. Pinheiro, Tsung-Yi Lin, Ronan Collobert, Piotr Dollàr
    * Institutions: [Facebook AI Research (FAIR)]
    * Abstract: Object segmentation requires both object-level information and low-level pixel data. This presents a challenge for feedforward networks: lower layers in convolutional nets capture rich spatial information, while upper layers encode object-level knowledge but are invariant to factors such as pose and appearance. In this work we propose to augment feedforward nets for object segmentation with a novel top-down refinement approach. The resulting bottom-up/top-down architecture is capable of efficiently generating high-fidelity object masks. Similarly to skip connections, our approach leverages features at all layers of the net. Unlike skip connections, our approach does not attempt to output independent predictions at each layer. Instead, we first output a coarse `mask encoding' in a feedforward pass, then refine this mask encoding in a top-down pass utilizing features at successively lower layers. The approach is simple, fast, and effective. Building on the recent DeepMask network for generating object proposals, we show accuracy improvements of 10-20% in average recall for various setups. Additionally, by optimizing the overall network architecture, our approach, which we call SharpMask, is 50% faster than the original DeepMask network (under .8s per image).
    * Comments:
        * > (2016, FPN) Similar architectures adopting top-down and skip connections are popular in recent research [28, 17, 8, 26]. Their goals are to produce a single high-level feature map of a fine resolution on which the predictions are to be made (Fig. 2 top).
* [[Instance-aware Semantic Segmentation via Multi-task Network Cascades](https://arxiv.org/abs/1512.04412)]
    [[pdf](https://arxiv.org/pdf/1512.04412.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1512.04412/)]
    * Title: Instance-aware Semantic Segmentation via Multi-task Network Cascades
    * Year: 14 Dec `2015`
    * Authors: Jifeng Dai, Kaiming He, Jian Sun
    * Abstract: Semantic segmentation research has recently witnessed rapid progress, but many leading methods are unable to identify object instances. In this paper, we present Multi-task Network Cascades for instance-aware semantic segmentation. Our model consists of three networks, respectively differentiating instances, estimating masks, and categorizing objects. These networks form a cascaded structure, and are designed to share their convolutional features. We develop an algorithm for the nontrivial end-to-end training of this causal, cascaded structure. Our solution is a clean, single-step training framework and can be generalized to cascades that have more stages. We demonstrate state-of-the-art instance-aware semantic segmentation accuracy on PASCAL VOC. Meanwhile, our method takes only 360ms testing an image using VGG-16, which is two orders of magnitude faster than previous systems for this challenging problem. As a by product, our method also achieves compelling object detection results which surpass the competitive Fast/Faster R-CNN systems. The method described in this paper is the foundation of our submissions to the MS COCO 2015 segmentation competition, where we won the 1st place.

### 9.2. Fully Convolutional Instance Segmentation (FCIS) Systems

* [[Instance-sensitive Fully Convolutional Networks](https://arxiv.org/abs/1603.08678)]
    [[pdf](https://arxiv.org/pdf/1603.08678.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1603.08678/)]
    * Title: Instance-sensitive Fully Convolutional Networks
    * Year: 29 Mar `2016`
    * Authors: Jifeng Dai, Kaiming He, Yi Li, Shaoqing Ren, Jian Sun
    * Abstract: Fully convolutional networks (FCNs) have been proven very successful for semantic segmentation, but the FCN outputs are unaware of object instances. In this paper, we develop FCNs that are capable of proposing instance-level segment candidates. In contrast to the previous FCN that generates one score map, our FCN is designed to compute a small set of instance-sensitive score maps, each of which is the outcome of a pixel-wise classifier of a relative position to instances. On top of these instance-sensitive score maps, a simple assembling module is able to output instance candidate at each position. In contrast to the recent DeepMask method for segmenting instances, our method does not have any high-dimensional layer related to the mask resolution, but instead exploits image local coherence for estimating instances. We present competitive results of instance segment proposal on both PASCAL VOC and MS COCO.
* [[Fully Convolutional Instance-aware Semantic Segmentation](https://arxiv.org/abs/1611.07709)]
    [[pdf](https://arxiv.org/pdf/1611.07709.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1611.07709/)]
    * Title: Fully Convolutional Instance-aware Semantic Segmentation
    * Year: 23 Nov `2016`
    * Authors: Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei
    * Institutions: [Tsinghua University], [Microsoft Research Asia]
    * Abstract: We present the first fully convolutional end-to-end solution for instance-aware semantic segmentation task. It inherits all the merits of FCNs for semantic segmentation and instance mask proposal. It performs instance mask prediction and classification jointly. The underlying convolutional representation is fully shared between the two sub-tasks, as well as between all regions of interest. The proposed network is highly integrated and achieves state-of-the-art performance in both accuracy and efficiency. It wins the COCO 2016 segmentation competition by a large margin. Code would be released at \url{this https URL}.

## 10. Multitask Learning (Panoptic Segmentation, 2018) (3)

* [[UberNet](https://arxiv.org/abs/1609.02132)]
    [[pdf](https://arxiv.org/pdf/1609.02132.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1609.02132/)]
    * Title: UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory
    * Year: 07 Sep `2016`
    * Authors: Iasonas Kokkinos
    * Institutions: CentraleSupélec - INRIA
    * Abstract: In this work we introduce a convolutional neural network (CNN) that jointly handles low-, mid-, and high-level vision tasks in a unified architecture that is trained end-to-end. Such a universal network can act like a `swiss knife' for vision tasks; we call this architecture an UberNet to indicate its overarching nature. We address two main technical challenges that emerge when broadening up the range of tasks handled by a single CNN: (i) training a deep architecture while relying on diverse training sets and (ii) training many (potentially unlimited) tasks with a limited memory budget. Properly addressing these two problems allows us to train accurate predictors for a host of tasks, without compromising accuracy. Through these advances we train in an end-to-end manner a CNN that simultaneously addresses (a) boundary detection (b) normal estimation (c) saliency estimation (d) semantic segmentation (e) human part segmentation (f) semantic boundary detection, (g) region proposal generation and object detection. We obtain competitive performance while jointly addressing all of these tasks in 0.7 seconds per frame on a single GPU. A demonstration of this system can be found at this http URL.
* [[The three R's of computer vision: Recognition, reconstruction and reorganization](https://www.sciencedirect.com/science/article/pii/S0167865516000313#:~:text=The%20three%20R's%20of%20computer%20vision%3A%20Recognition%2C%20reconstruction%20and%20reorganization%E2%98%86)]
    [[pdf](https://pdf.sciencedirectassets.com/271524/1-s2.0-S0167865516X00050/1-s2.0-S0167865516000313/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEcaCXVzLWVhc3QtMSJIMEYCIQCGiTYkl8Mxv0WCS%2FiGMO%2FEe7PNyAcZtmHR5lV4ekMHEwIhANBhC0q3cwI%2FC8KHuig64KzNlYtluIjE8pOzLNodsgpgKtsECMD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1Igz7NVJSw%2Fgy99oMSb4qrwS5mLne6JqoJmtmvcuBr7nIrrMKnd4ZR1EQnUvgBzJYPf9l%2FUqtVAcqZAqNiPpDWfViXBkbUPmAasB6f3t400A88TKaxIJnNgz%2FwvpP8IK4NoGCzrlUYYi8m0E9Ju1LeR2%2B6iAXNfug4Lxh03IkrTtpXypsUCaTKh5UExEZNHSjBi8SvNcWVapFh5UeFKq4AbvBO2OhS5Ijp2Osb512abzI0YROW9%2FRQ%2FSHyLlrbJBzRYWVEOYer%2Bqe3OglEdfuhwsV9cjXcZKCsZU7HDBYIjRVMcmhvY7KyFZA74DxiJgDW%2FvFAaSh%2Bp4KB%2BU%2FOCgBhJCAJF%2BrAeySyBC30whIQR4sBigXop5dMavarsViSh%2FBHbovejxlza%2Bj5j%2Fi62qxcA2yTVKiDCCKrtyhxzX8z%2F3If1Qsanht1QUZ%2FdTTX3d6P15nEYSKncMJ8GikfsXb7TZTmoe9AdZEWOU%2FG1fMaNXyveE4cuBsDBrj9d78TSTMUYbsASdMXLcFk6L88b2ZSQ%2BHGlyduNwW96aDMix6WbiSsaBflpr6wV%2Bsbmjv9gu0mD%2FovjzkS%2F3WMmWgum5eUjKChkkvgZCwmbTSXhEZvfnqMlR%2B0AvTCsUaEogbGlvwK4P5b2HICPJSuRcCp86NFuFVG7%2B6u9td9NVsMnSP0Tm%2Bm1wehGrObx3N62QKYNeuGgBxCkUeSdH95hRM5Mqlk%2F%2Fx%2FErwn88nUZWPAY4AF3CVolmBaF1fkxwatpyUUiUGMJKSnpgGOqgBTm3SCoeihiNFQfe4jSh94q%2B4ATOKqK7kw74Xp9NOoG3J8Pjq6JW84Y4wAYhRvGaGPr7vtncX%2FbQdq1v1%2FN%2FdgHfm0lYe1YXCHX2XrTlPMNEXSOsR0FoD7UmSYRooVNp7f8%2FjVoZOfqe3gr9hPvrjbjrBvGOJmBDl2prfxtwPOu6uMY9DpZgQ43seCL6Z2AfU0zkOSPPE4f0HxWHAzsKc71OyNvso7gA5&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220825T154849Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2RO2MC73%2F20220825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=5c5f1475a6c2a82e24a5d914fb0f86d56b98768aa13147f9fdc96486b0db52b1&hash=6622a043fc37276619ed095b5b2e06e3ca76767da78e589ef75b5ad313b9130b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0167865516000313&tid=spdf-e1153792-4079-43b2-98a0-827714bebb86&sid=943da4f77804714c1b2a006928b3096a7eeegxrqa&type=client&ua=4d52555a555b55500603&rr=74057960ef153ffd)]
    * Title: The three R's of computer vision: Recognition, reconstruction and reorganization
    * Year: 29 June `2015`
    * Authors: J. Malik, P. Arbeláez, J. Carreira, K. Fragkiadaki, R. Girshick, G. Gkioxari, S. Gupta, B. Hariharan, A. Kar, and S. Tulsiani
    * Institutions: [EECS, UC Berkeley, Berkeley], [Universidad de los Andes], [Facebook]
    * Abstract: We argue for the importance of the interaction between recognition, reconstruction and re-organization, and propose that as a unifying framework for computer vision. In this view, recognition of objects is reciprocally linked to re-organization, with bottom-up grouping processes generating candidates, which can be classified using top down knowledge, following which the segmentations can be refined again. Recognition of 3D objects could benefit from a reconstruction of 3D structure, and 3D reconstruction can benefit from object category-specific priors. We also show that reconstruction of 3D structure from video data goes hand in hand with the reorganization of the scene. We demonstrate pipelined versions of two systems, one for RGB-D images, and another for RGB images, which produce rich 3D scene interpretations in this framework.
* [[Cross-stitch Networks for Multi-task Learning](https://arxiv.org/abs/1604.03539)]
    [[pdf](https://arxiv.org/pdf/1604.03539.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1604.03539/)]
    * Title: Cross-stitch Networks for Multi-task Learning
    * Year: 12 Apr `2016`
    * Authors: Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert
    * Institutions: [The Robotics Institute, Carnegie Mellon University]
    * Abstract: Multi-task learning in Convolutional Networks has displayed remarkable success in the field of recognition. This success can be largely attributed to learning shared representations from multiple supervisory tasks. However, existing multi-task approaches rely on enumerating multiple network architectures specific to the tasks at hand, that do not generalize. In this paper, we propose a principled approach to learn shared representations in ConvNets using multi-task learning. Specifically, we propose a new sharing unit: "cross-stitch" unit. These units combine the activations from multiple networks and can be trained end-to-end. A network with cross-stitch units can learn an optimal combination of shared and task-specific representations. Our proposed method generalizes across multiple tasks and shows dramatically improved performance over baseline methods for categories with few training examples.

## 11. Panoptic Segmentation

* [[Panoptic Segmentation](https://arxiv.org/abs/1801.00868)]
    [[pdf](https://arxiv.org/pdf/1801.00868.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1801.00868/)]
    * Title: Panoptic Segmentation
    * Year: 03 Jan `2018`
    * Authors: Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, Piotr Dollár
    * Institutions: [Facebook AI Research (FAIR)], [HCI/IWR, Heidelberg University, Germany]
    * Abstract: We propose and study a task we name panoptic segmentation (PS). Panoptic segmentation unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance). The proposed task requires generating a coherent scene segmentation that is rich and complete, an important step toward real-world vision systems. While early work in computer vision addressed related image/scene parsing tasks, these are not currently popular, possibly due to lack of appropriate metrics or associated recognition challenges. To address this, we propose a novel panoptic quality (PQ) metric that captures performance for all classes (stuff and things) in an interpretable and unified manner. Using the proposed metric, we perform a rigorous study of both human and machine performance for PS on three existing datasets, revealing interesting insights about the task. The aim of our work is to revive the interest of the community in a more unified view of image segmentation.
* [[Panoptic FPN](https://arxiv.org/abs/1901.02446)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1901.02446.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1901.02446/)]
    * Title: Panoptic Feature Pyramid Networks
    * Year: 08 Jan `2019`
    * Authors: Alexander Kirillov, Ross Girshick, Kaiming He, Piotr Dollár
    * Institutions: [Facebook AI Research (FAIR)]
    * Abstract: The recently introduced panoptic segmentation task has renewed our community's interest in unifying the tasks of instance segmentation (for thing classes) and semantic segmentation (for stuff classes). However, current state-of-the-art methods for this joint task use separate and dissimilar networks for instance and semantic segmentation, without performing any shared computation. In this work, we aim to unify these methods at the architectural level, designing a single network for both tasks. Our approach is to endow Mask R-CNN, a popular instance segmentation method, with a semantic segmentation branch using a shared Feature Pyramid Network (FPN) backbone. Surprisingly, this simple baseline not only remains effective for instance segmentation, but also yields a lightweight, top-performing method for semantic segmentation. In this work, we perform a detailed study of this minimally extended version of Mask R-CNN with FPN, which we refer to as Panoptic FPN, and show it is a robust and accurate baseline for both tasks. Given its effectiveness and conceptual simplicity, we hope our method can serve as a strong baseline and aid future research in panoptic segmentation.
    * Comments:
        * > (2021, PVT) Kirillov et al. [21] develop a lightweight segmentation head termed Semantic FPN, based on FPN [26].

## 12. Scene Parsing

* [[Deep Convolutional Networks for Scene Parsing](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.183.8571)]
    [[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.183.8571&rep=rep1&type=pdf)]
    * Title: Deep Convolutional Networks for Scene Parsing
    * Year: `2009`
    * Authors: David Grangier , Léon Bottou , Ronan Collobert
    * Abstract: We propose a deep learning strategy for scene parsing, i.e. to asssign a class label to each pixel of an image. We investigate the use of deep convolutional network for modeling the complex scene label structures, relying on a supervised greedy learning strategy. Compared to standard approaches based on CRFs, our strategy does not need hand-crafted features, allows modeling more complex spatial dependencies and has a lower inference cost. Experiments over the MSRC benchmark and the LabelMe dataset show the effectiveness of our approach.
* [[Deep Deconvolutional Networks for Scene Parsing](https://arxiv.org/abs/1411.4101)]
    [[pdf](https://arxiv.org/pdf/1411.4101.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1411.4101/)]
    * Title: Deep Deconvolutional Networks for Scene Parsing
    * Year: 15 Nov `2014`
    * Authors: Rahul Mohan
    * Abstract: Scene parsing is an important and challenging prob- lem in computer vision. It requires labeling each pixel in an image with the category it belongs to. Tradition- ally, it has been approached with hand-engineered features from color information in images. Recently convolutional neural networks (CNNs), which automatically learn hierar- chies of features, have achieved record performance on the task. These approaches typically include a post-processing technique, such as superpixels, to produce the final label- ing. In this paper, we propose a novel network architecture that combines deep deconvolutional neural networks with CNNs. Our experiments show that deconvolutional neu- ral networks are capable of learning higher order image structure beyond edge primitives in comparison to CNNs. The new network architecture is employed for multi-patch training, introduced as part of this work. Multi-patch train- ing makes it possible to effectively learn spatial priors from scenes. The proposed approach yields state-of-the-art per- formance on four scene parsing datasets, namely Stanford Background, SIFT Flow, CamVid, and KITTI. In addition, our system has the added advantage of having a training system that can be completely automated end-to-end with- out requiring any post-processing.
* [Scene Parsing with Object Instances and Occlusion Ordering](https://ieeexplore.ieee.org/document/6909874)
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6909874)]
    * Title: Scene Parsing with Object Instances and Occlusion Ordering
    * Year: 25 September `2014`
    * Authors: Joseph Tighe; Marc Niethammer; Svetlana Lazebnik
    * Institutions: [University of North Carolina at Chapel Hill], [University of Illinois at Urbana-Champaign]
    * Abstract: This work proposes a method to interpret a scene by assigning a semantic label at every pixel and inferring the spatial extent of individual object instances together with their occlusion relationships. Starting with an initial pixel labeling and a set of candidate object masks for a given test image, we select a subset of objects that explain the image well and have valid overlap relationships and occlusion ordering. This is done by minimizing an integer quadratic program either using a greedy method or a standard solver. Then we alternate between using the object predictions to refine the pixel labels and vice versa. The proposed system obtains promising results on two challenging subsets of the LabelMe and SUN datasets, the largest of which contains 45, 676 images and 232 classes.
* [[PSPNet](https://arxiv.org/abs/1612.01105)]
    [[pdf](https://arxiv.org/pdf/1612.01105.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1612.01105/)]
    * Title: Pyramid Scene Parsing Network
    * Year: 04 Dec `2016`
    * Authors: Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia
    * Institutions: [The Chinese University of Hong Kong], [SenseTime Group Limited]
    * Abstract: Scene parsing is challenging for unrestricted open vocabulary and diverse scenes. In this paper, we exploit the capability of global context information by different-region-based context aggregation through our pyramid pooling module together with the proposed pyramid scene parsing network (PSPNet). Our global prior representation is effective to produce good quality results on the scene parsing task, while PSPNet provides a superior framework for pixel-level prediction tasks. The proposed approach achieves state-of-the-art performance on various datasets. It came first in ImageNet scene parsing challenge 2016, PASCAL VOC 2012 benchmark and Cityscapes benchmark. A single PSPNet yields new record of mIoU accuracy 85.4% on PASCAL VOC 2012 and accuracy 80.2% on Cityscapes.
    * Comments:
        * > (2019, Panoptic FPN) To increase feature resolution, which is necessary for generating high-quality results, recent top methods [12, 56, 5, 57] rely heavily on the use of dilated convolution [55] (also known as atrous convolution [10]). While effective, such an approach can substantially increase compute and memory, limiting the type of backbone network that can be used.
        * > (2021, PVT) Zhao et al. [62] designs a pyramid pooling module over various pooling scales.
* [[PSANet](https://link.springer.com/chapter/10.1007/978-3-030-01240-3_17)]
    [[pdf](https://link.springer.com/content/pdf/10.1007/978-3-030-01240-3_17.pdf)]
    * Title: PSANet: Point-wise Spatial Attention Network for Scene Parsing
    * Year: 05 Oct `2018`
    * Authors: Hengshuang Zhao, Yi Zhang, Shu Liu, Jianping Shi, Chen Change Loy, Dahua Lin & Jiaya Jia
    * Institutions: [The Chinese University of Hong Kong, Shatin, Hong Kong], [CUHK-Sensetime Joint Lab, The Chinese University of Hong Kong, Shatin, Hong Kong], [SenseTime Research, Beijing , China], [Nanyang Technological University, Singapore, Singapore], [Tencent Youtu Lab, Shenzhen, China]
    * Abstract: We notice information flow in convolutional neural networks is restricted inside local neighborhood regions due to the physical design of convolutional filters, which limits the overall understanding of complex scenes. In this paper, we propose the point-wise spatial attention network (PSANet) to relax the local neighborhood constraint. Each position on the feature map is connected to all the other ones through a self-adaptively learned attention mask. Moreover, information propagation in bi-direction for scene parsing is enabled. Information at other positions can be collected to help the prediction of the current position and vice versa, information at the current position can be distributed to assist the prediction of other ones. Our proposed approach achieves top performance on various competitive scene parsing datasets, including ADE20K, PASCAL VOC 2012 and Cityscapes, demonstrating its effectiveness and generality.
    * Comments:
        * > To increase feature resolution, which is necessary for generating high-quality results, recent top methods [12, 56, 5, 57] rely heavily on the use of dilated convolution [55] (also known as atrous convolution [10]). While effective, such an approach can substantially increase compute and memory, limiting the type of backbone network that can be used. (Panoptic FPN, 2019)

## 13. weakly supervised (2015, DeconvNet)

* [[BoxSup](https://arxiv.org/abs/1503.01640)]
    [[pdf](https://arxiv.org/pdf/1503.01640.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1503.01640/)]
    * Title: BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation
    * Year: 05 Mar `2015`
    * Authors: Jifeng Dai, Kaiming He, Jian Sun
    * Abstract: Recent leading approaches to semantic segmentation rely on deep convolutional networks trained with human-annotated, pixel-level segmentation masks. Such pixel-accurate supervision demands expensive labeling effort and limits the performance of deep networks that usually benefit from more training data. In this paper, we propose a method that achieves competitive accuracy but only requires easily obtained bounding box annotations. The basic idea is to iterate between automatically generating region proposals and training convolutional networks. These two steps gradually recover segmentation masks for improving the networks, and vise versa. Our method, called BoxSup, produces competitive results supervised by boxes only, on par with strong baselines fully supervised by masks under the same setting. By leveraging a large amount of bounding boxes, BoxSup further unleashes the power of deep convolutional networks and yields state-of-the-art results on PASCAL VOC 2012 and PASCAL-CONTEXT.
    * Comments:
        * > (2015, DeconvNet) When only bounding box annotations are given for input images, [2, 19] refine the annotations through iterative procedures and obtain accurate segmentation outputs.
* [Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation]
    * Title: Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation
    * Comments:
        * > (2015, DeconvNet) When only bounding box annotations are given for input images, [2, 19] refine the annotations through iterative procedures and obtain accurate segmentation outputs.
* [From Image-level to Pixel-level Labeling with Convolutional Networks]
    * Title: From Image-level to Pixel-level Labeling with Convolutional Networks
    * Comments:
        * > (2015, DeconvNet) On the other hand, [20] performs semantic segmentation based only on image-level annotations in a multiple instance learning framework.
