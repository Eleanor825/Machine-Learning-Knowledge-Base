#! https://zhuanlan.zhihu.com/p/558646271
# [Notes][Vision][Segmentation] DeconvNet

* url: https://arxiv.org/abs/1505.04366
* Title: Learning Deconvolution Network for Semantic Segmentation
* Year: 17 May `2015`
* Authors: Hyeonwoo Noh, Seunghoon Hong, Bohyung Han
* Institutions: [Department of Computer Science and Engineering, POSTECH, Korea]
* Abstract: We propose a novel semantic segmentation algorithm by learning a deconvolution network. We learn the network on top of the convolutional layers adopted from VGG 16-layer net. The deconvolution network is composed of deconvolution and unpooling layers, which identify pixel-wise class labels and predict segmentation masks. We apply the trained network to each proposal in an input image, and construct the final semantic segmentation map by combining the results from all proposals in a simple manner. The proposed algorithm mitigates the limitations of the existing methods based on fully convolutional networks by integrating deep deconvolution network and proposal-wise prediction; our segmentation method typically identifies detailed structures and handles objects in multiple scales naturally. Our network demonstrates outstanding performance in PASCAL VOC 2012 dataset, and we achieve the best accuracy (72.5%) among the methods trained with no external data through ensemble with the fully convolutional network.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> Semantic segmentation based on FCNs [1, 17] have a couple of critical limitations.

> First, the network can handle only a single scale semantics within image due to the fixed-size receptive field. Therefore, the object that is substantially larger or smaller than the receptive field may be fragmented or mislabeled. In other words, label prediction is done with only local information for large objects and the pixels that belong to the same object may have inconsistent labels as shown in Figure 1(a). Also, small objects are often ignored and classified as background, which is illustrated in Figure 1(b). Although [17] attempts to sidestep this limitation using skip architecture, this is not a fundamental solution and performance gain is not significant.

> Second, the detailed structures of an object are often lost or smoothed because the label map, input to the deconvolutional layer, is too coarse and deconvolution procedure is overly simple. Note that, in the original FCN [17], the label map is only 16x16 in size and is deconvolved to generate segmentation result in the original input size through bilinear interpolation. The absence of real deconvolution in [1, 17] makes it difficult to achieve good performance. However, recent methods ameliorate this problem using CRF [14].

> To overcome such limitations, we employ a completely different strategy to perform semantic segmentation based on CNN. Our main contributions are summarized below:
> * We learn a multi-layer deconvolution network, which is composed of deconvolution, unpooling, and rectified linear unit (ReLU) layers. Learning deconvolution network for semantic segmentation is meaningful but no one has attempted to do it yet to our knowledge.
> * The trained network is applied to individual object proposals to obtain instance-wise segmentations, which are combined for the final semantic segmentation; it is free from scale issues found in FCN-based methods and identifies finer details of an object.
> * We achieve outstanding performance using the deconvolution network trained only on PASCAL VOC 2012 dataset, and obtain the best accuracy through the ensemble with [17] by exploiting the heterogeneous and complementary characteristics of our algorithm with respect to FCN-based methods.

## 2. Related Work

## 3. System Architecture

### 3.1. Architecture

> Our trained network is composed of two parts—convolution and deconvolution networks.

> The convolution network corresponds to `feature extractor` that transforms the input image to multidimensional feature representation, whereas the deconvolution network is a `shape generator` that produces object segmentation from the feature extracted from the convolution network.

> The final output of the network is a probability map in the same size to input image, indicating probability of each pixel that belongs to one of the predefined classes.

### 3.2. Deconvolution Network for Segmentation

#### 3.2.1 Unpooling

> Pooling in convolution network is designed to filter noisy activations in a lower layer by abstracting activations in a receptive field with a single representative value.

> Although it helps classification by retaining only robust activations in upper layers, spatial information within a receptive field is lost during pooling, which may be critical for precise localization that is required for semantic segmentation.

> To implement the unpooling operation, we follow the similar approach proposed in [24, 25]. It records the locations of maximum activations selected during pooling operation in switch variables, which are employed to place each activation back to its original pooled location. This unpooling strategy is particularly useful to reconstruct the structure of input object as described in [24].

#### 3.2.2 Deconvolution

> The output of an unpooling layer is an enlarged, yet `sparse` activation map. The deconvolution layers `densify` the sparse activations obtained by unpooling through convolution-like operations with multiple learned filters.

> However, contrary to convolutional layers, which connect multiple input activations within a filter window to a single activation, deconvolutional layers associate a single input activation with multiple outputs.

> The learned filters in deconvolutional layers correspond to bases to reconstruct `shape` of an input object.

> Therefore, similar to the convolution network, a `hierarchical` structure of deconvolutional layers are used to capture different level of shape details. The filters in lower layers tend to capture overall shape of an object while the class-specific fine-details are encoded in the filters in higher layers. In this way, the network directly takes class-specific shape information into account for semantic segmentation, which is often ignored in other approaches based only on convolutional layers [1, 17].

#### 3.2.3 Analysis of Deconvolution Network

> Note that unpooling and deconvolution play different roles for the construction of segmentation masks.

> Unpooling captures example-specific structures by tracing the original locations with strong activations back to image space. As a result, it effectively reconstructs the detailed structure of an object in finer resolutions.

> On the other hand, learned filters in deconvolutional layers tend to capture class-specific shapes. Through deconvolutions, the activations closely related to the target classes are amplified while noisy activations from other regions are suppressed effectively.

### 3.3. System Overview

> Our algorithm poses semantic segmentation as instance-wise segmentation problem. That is, the network takes a sub-image potentially containing objects—which we refer to as instance(s) afterwards—as an input and produces pixel-wise class prediction as an output. Given our network, semantic segmentation on a whole image is obtained by applying the network to each candidate proposals extracted from the image and aggregating outputs of all proposals to the original image space.

## 4. Training

### 4.1. Batch Normalization

> We perform the batch normalization [11] to reduce the internal-covariate-shift by normalizing input distributions of every layer to the standard Gaussian distribution. For the purpose, a batch normalization layer is added to the output of every convolutional and deconvolutional layer. We observe that the batch normalization is critical to optimize our network; it ends up with a poor local optimum without batch normalization.

### 4.2. Two-stage Training

> We employ a two-stage training method to address this issue, where we train the network with easy examples first and fine-tune the trained network with more challenging examples later.

> To construct training examples for the first stage training, we crop object instances using ground-truth annotations so that an object is centered at the cropped bounding box. By limiting the variations in object location and size, we reduce search space for semantic segmentation significantly and train the network with much less training examples successfully.

> In the second stage, we utilize object proposals to construct more challenging examples. Specifically, candidate proposals sufficiently overlapped with ground-truth segmentations are selected for training. Using the proposals to construct training data makes the network more robust to the misalignment of proposals in testing, but makes training more challenging since the location and scale of an object may be significantly different across training examples.

## 5. Inference

> The proposed network is trained to perform semantic segmentation for individual instances. Given an input image, we first generate a sufficient number of candidate proposals, and apply the trained network to obtain semantic segmentation maps of individual proposals. Then we aggregate the outputs of all proposals to produce semantic segmentation on a whole image.

### 5.1. Aggregating Instance-wise Segmentation Maps

Notations:
* Let $C \in \mathbb{Z}_{++}$ denote the number of classes.
* Let $H, W \in \mathbb{Z}_{++}$ denote the height and width of the output score maps.
* Let $g_{i} \in \mathbb{R}^{H \times W \times C}$ denote the output score map of the $i$-th proposal.
* Let $G_{i}$ denote the segmentation map obtained by padding $g_{i}$ to the original image size.

> Then we construct the pixel-wise class score map of an image by aggregating the outputs of all proposals by
$$P(x, y, c) := \max_{i}G_{i}(x, y, c) \tag{1}$$
of
$$P(x, y, c) := \sum_{i}G_{i}(x, y, c). \tag{2}$$

> Class conditional probability maps in the original image space are obtained by applying softmax function to the aggregated maps obtained by Eq. (1) or (2).

> Finally, we apply the fully-connected CRF [14] to the output maps for the final pixel-wise labeling, where unary potential are obtained from the pixel-wise class conditional probability maps.

### 5.2. Ensemble with FCN

> Our algorithm based on the deconvolution network has complementary characteristics to the approaches relying on FCN.

> Our deconvolution network is appropriate to capture the fine-details of an object, whereas FCN is typically good at extracting the overall shape of an object.

> In addition, instance-wise prediction is useful for handling objects with various scales, while fully convolutional network with a coarse scale may be advantageous to capture context within image.

> We develop a simple method to combine the outputs of both algorithms. Given two sets of class conditional probability maps of an input image computed independently by the proposed method and FCN, we compute the `mean` of both output maps and apply the CRF to obtain the final semantic segmentation.

## 6. Experiments

## 7. Conclusion

----------------------------------------------------------------------------------------------------

## References

* Noh, Hyeonwoo, Seunghoon Hong, and Bohyung Han. "Learning deconvolution network for semantic segmentation." *Proceedings of the IEEE international conference on computer vision*. 2015.

## Further Reading

* [1] DeepLabv1
* [7] R-CNN
* [11] Inception-v2/Batch Normalization
* [15] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [17] [Fully Convolutional Networks (FCN)](https://zhuanlan.zhihu.com/p/561031110)
* [22] [VGGNet](https://zhuanlan.zhihu.com/p/563314926)
* [23] [Inception-v1/GoogLeNet](https://zhuanlan.zhihu.com/p/564141144)
