# [Notes][Vision][Segmentation] SegNet

* url: https://arxiv.org/abs/1511.00561
* Title: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
* Year: 02 Nov `2015`
* Authors: Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
* Institutions: [Machine Intelligence Lab, Department of Engineering, University of Cambridge, UK]
* Abstract: We present a novel and practical deep fully convolutional neural network architecture for semantic pixel-wise segmentation termed SegNet. This core trainable segmentation engine consists of an encoder network, a corresponding decoder network followed by a pixel-wise classification layer. The architecture of the encoder network is topologically identical to the 13 convolutional layers in the VGG16 network. The role of the decoder network is to map the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification. The novelty of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature map(s). Specifically, the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling. This eliminates the need for learning to upsample. The upsampled maps are sparse and are then convolved with trainable filters to produce dense feature maps. We compare our proposed architecture with the widely adopted FCN and also with the well known DeepLab-LargeFOV, DeconvNet architectures. This comparison reveals the memory versus accuracy trade-off involved in achieving good segmentation performance. SegNet was primarily motivated by scene understanding applications. Hence, it is designed to be efficient both in terms of memory and computational time during inference. It is also significantly smaller in the number of trainable parameters than other competing architectures. We also performed a controlled benchmark of SegNet and other architectures on both road scenes and SUN RGB-D indoor scene segmentation tasks. We show that SegNet provides good performance with competitive inference time and more efficient inference memory-wise as compared to other architectures. We also provide a Caffe implementation of SegNet and a web demo at [this http URL](http://mi.eng.cam.ac.uk/projects/segnet/).

----------------------------------------------------------------------------------------------------

## 1 INTRODUCTION

> The encoder network in SegNet is topologically identical to the convolutional layers in VGG16 [1].

> The key component of SegNet is the decoder network which consists of a hierarchy of decoders one corresponding to each encoder.

## 2 LITERATURE REVIEW

## 3 ARCHITECTURE

> SegNet has an encoder network and a corresponding decoder network, followed by a final pixelwise classification layer.

> The encoder network consists of 13 convolutional layers which correspond to the first 13 convolutional layers in the VGG16 network [1] designed for object classification.

> Each encoder layer has a corresponding decoder layer and hence the decoder network has 13 layers.

> The final decoder output is fed to a multi-class soft-max classifier to produce class probabilities for each pixel independently.

Encoder

* Conv
* BatchNormalization
* ReLU
* MaxPool(size=(2, 2), strides=(2, 2))

> While several layers of max-pooling and sub-sampling can achieve more translation invariance for robust classification correspondingly there is a loss of spatial resolution of the feature maps.

> The increasingly lossy (boundary detail) image representation is not beneficial for segmentation where boundary delineation is vital. Therefore, it is necessary to capture and store boundary information in the encoder feature maps before sub-sampling is performed.

Decoder

> The appropriate decoder in the decoder network upsamples its input feature map(s) using the memorized max-pooling indices from the corresponding encoder feature map(s). This step produces sparse feature map(s).

> These feature maps are then convolved with a trainable decoder filter bank to produce dense feature maps.

> A batch normalization step is then applied to each of these maps.

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
* [4] DeconvNet
* [5] Inception-v1/GoogLeNet
* [6] VGG
* [15] Dilated Convolutions
* [16] U-Net
* [51] Inception-v2/Batch Normalization
* [53] DeconvNet
* [58] Fully Convolutional Networks (FCN)
