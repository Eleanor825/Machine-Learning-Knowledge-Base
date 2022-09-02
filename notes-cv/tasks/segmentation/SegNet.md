# [Notes][Vision][Segmentation] SegNet

* url: https://arxiv.org/abs/1511.00561
* Title: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
* Year: 02 Nov `2015`
* Authors: Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
* Institutions: [Machine Intelligence Lab, Department of Engineering, University of Cambridge, UK]
* Abstract: We present a novel and practical deep fully convolutional neural network architecture for semantic pixel-wise segmentation termed SegNet. This core trainable segmentation engine consists of an encoder network, a corresponding decoder network followed by a pixel-wise classification layer. The architecture of the encoder network is topologically identical to the 13 convolutional layers in the VGG16 network. The role of the decoder network is to map the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification. The novelty of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature map(s). Specifically, the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling. This eliminates the need for learning to upsample. The upsampled maps are sparse and are then convolved with trainable filters to produce dense feature maps. We compare our proposed architecture with the widely adopted FCN and also with the well known DeepLab-LargeFOV, DeconvNet architectures. This comparison reveals the memory versus accuracy trade-off involved in achieving good segmentation performance. SegNet was primarily motivated by scene understanding applications. Hence, it is designed to be efficient both in terms of memory and computational time during inference. It is also significantly smaller in the number of trainable parameters than other competing architectures. We also performed a controlled benchmark of SegNet and other architectures on both road scenes and SUN RGB-D indoor scene segmentation tasks. We show that SegNet provides good performance with competitive inference time and more efficient inference memory-wise as compared to other architectures. We also provide a Caffe implementation of SegNet and a web demo at [this http URL](http://mi.eng.cam.ac.uk/projects/segnet/).

----------------------------------------------------------------------------------------------------

## 1 INTRODUCTION

> Our motivation to design SegNet arises from this need to map low resolution features to input resolution for pixel-wise classification. This mapping must produce features which are useful for accurate boundary localization.

> The `encoder` network in SegNet is topologically identical to the convolutional layers in VGG16 [1].

> The key component of SegNet is the `decoder` network which consists of a hierarchy of decoders one corresponding to each encoder.

> Reusing max-pooling indices in the decoding process has several practical advantages;
> 1. (i) it improves boundary delineation,
> 2. (ii) it reduces the number of parameters enabling end-to-end training, and
> 3. (iii) this form of upsampling can be incorporated into any encoder-decoder architecture such as [2, 10] with only a little modification.

## 2 LITERATURE REVIEW

Encoder-Decoder Architecture

> Newer deep architectures [2, 4, 13, 18, 10] particularly designed for segmentation have advanced the state-of-the-art by learning to decode or map low resolution image representations to pixel-wise predictions.

> The `encoder` network which produces these low resolution representations in all of these architectures is the VGG16 classification network [1] which has 13 convolutional layers and 3 fully connected layers.

> The `decoder` network varies between these architectures and is the part which is responsible for producing multi-dimensional features for each pixel for classification.

Multi-Scale Architecture

> Multi-scale deep architectures are also being pursued [13, 44]. They come in two flavours,
> 1. (i) those which use input images at a few scales and corresponding deep feature extraction networks, and
> 2. (ii) those which combine feature maps from different layers of a single deep architecture [45] [11].

> The common idea is to use features extracted at multiple scales to provide both local and global context [46] and the using feature maps of the early encoding layers retain more high frequency detail leading to sharper class boundaries.

Others

> Several of the recently proposed deep architectures for segmentation are not feed-forward in inference time [4], [3], [18]. They require either MAP inference over a CRF [44], [43] or aids such as region proposals [4] for inference.

> We believe the perceived performance increase obtained by using a CRF is due to the lack of good decoding techniques in their core feed-forward segmentation engine.

> In this work we discard the fully connected layers of the VGG16 encoder network which enables us to train the network using the relevant training set using SGD optimization.

> Here, SegNet differs from these architectures as the deep encoder-decoder network is trained jointly for a supervised learning task and hence the decoders are an integral part of the network in test time.

## 3 ARCHITECTURE

Overview

> SegNet has an encoder network and a corresponding decoder network, followed by a final pixelwise classification layer.

> The encoder network consists of 13 convolutional layers which correspond to the first 13 convolutional layers in the VGG16 network [1] designed for object classification.

> Each encoder layer has a corresponding decoder layer and hence the decoder network has 13 layers.

> The final decoder output is fed to a multi-class soft-max classifier to produce class probabilities for each pixel independently.

Encoder

> Each encoder in the `encoder network` performs convolution with a filter bank to produce a set of feature maps.

* Conv
* BatchNormalization
* ReLU
* MaxPool(size=(2, 2), strides=(2, 2))

> While several layers of max-pooling and sub-sampling can achieve more translation invariance for robust classification correspondingly there is a loss of spatial resolution of the feature maps.

> The increasingly lossy (boundary detail) image representation is not beneficial for segmentation where boundary delineation is vital. Therefore, it is necessary to capture and store boundary information in the encoder feature maps before sub-sampling is performed.

Decoder

> The appropriate decoder in the `decoder network` upsamples its input feature map(s) using the memorized max-pooling indices from the corresponding encoder feature map(s).

> This step produces sparse feature map(s).

> These feature maps are then convolved with a trainable decoder filter bank to produce dense feature maps.

> Note that the decoder corresponding to the first encoder (closest to the input image) produces a multi-channel feature map, although its encoder input has 3 channels (RGB). This is unlike the other decoders in the network which produce feature maps with the same number of size and channels as their encoder inputs.

Classifier

> The output of the soft-max classifier is a K channel image of probabilities where K is the number of classes.

### 3.1 Decoder Variants

### 3.2 Training

### 3.3 Analysis

## 4 BENCHMARKING

## 5 DISCUSSION AND FUTURE WORK

## 6 CONCLUSION

----------------------------------------------------------------------------------------------------

## References

* Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." *IEEE transactions on pattern analysis and machine intelligence* 39.12 (2017): 2481-2495.

## Further Reading

* [1] VGG
* [2] Fully Convolutional Networks (FCN)
* [3] DeepLabv1
* [4] [DeconvNet](https://zhuanlan.zhihu.com/p/558646271)
* [5] Inception-v1/GoogLeNet
* [6] VGG
* [15] Dilated Convolutions
* [16] U-Net
* [51] Inception-v2/Batch Normalization
* [53] [DeconvNet](https://zhuanlan.zhihu.com/p/558646271)
* [58] Fully Convolutional Networks (FCN)
