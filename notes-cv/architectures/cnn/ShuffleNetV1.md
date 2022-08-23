# [Notes][Vision][CNN] ShuffleNet V1

* url: https://arxiv.org/abs/1707.01083
* Title: ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
* Year: 04 Jul `2017`
* Authors: Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
* Abstract: We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet on ImageNet classification task, under the computation budget of 40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves ~13x actual speedup over AlexNet while maintaining comparable accuracy.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> To overcome the side effects brought by group convolutions, we come up with a novel channel shuffle operation to help the information flowing across feature channels.

> Compared with popular structures like [30, 9, 40], for a given computation complexity budget, our ShuffleNet allows more feature map channels, which helps to encode more information and is especially critical to the performance of very small networks.

## 2. Related Work

**Efficient Model Designs**

**Group Convolution**

**Channel Shuffle Operation**

**Model Acceleration**

## 3. Approach

### 3.1. Channel Shuffle for Group Convolutions

> State-of-the-art networks such as Xception [3] and ResNeXt [40] introduce efficient depthwise separable convolutions or group convolutions into the building blocks to strike an excellent trade-off between representation capability and computational cost. However, we notice that both designs do not fully take the 1 x 1 convolutions (also called pointwise convolutions in [12]) into account, which require considerable complexity.

> By ensuring that each convolution operates only on the corresponding input channel group, group convolution significantly reduces computation cost. However, if multiple group convolutions stack together, there is one side effect: outputs from a certain channel are only derived from a small fraction of input channels.

> This property blocks information flow between channel groups and weakens representation.

> If we allow group convolution to obtain input data from different groups (as shown in Fig 1 (b)), the input and output channels will be fully related. Specifically, for the feature map generated from the previous group layer, we can first divide the channels in each group into several subgroups, then feed each group in the next layer with different subgroups.

> This can be efficiently and elegantly implemented by a channel shuffle operation (Fig 1 (c)): suppose a convolutional layer with g groups whose output has g x n channels; we first reshape the output channel dimension into (g, n), transposing and then flattening it back as the input of next layer.

> Note that the operation still takes effect even if the two convolutions have different numbers of groups.

> Moreover, channel shuffle is also differentiable, which means it can be embedded into network structures for end-to-end training.

### 3.2. ShuffleNet Unit

> We start from the design principle of bottleneck unit [9] in Fig 2 (a). It is a residual block. In its residual branch, for the 3 x 3 layer, we apply a computational economical 3 x 3 depthwise convolution [3] on the bottleneck feature map. Then, we replace the first 1 x 1 layer with pointwise group convolution followed by a channel shuffle operation, to form a ShuffleNet unit, as shown in Fig 2 (b).

> As for the case where ShuffleNet is applied with stride, we simply make two modifications (see Fig 2 (c)): (i) add a 3 x 3 average pooling on the shortcut path; (ii) replace the element-wise addition with channel concatenation, which makes it easy to enlarge channel dimension with little extra computation cost.

> Given a computational budget, ShuffleNet can use wider feature maps. We find this is critical for small networks, as tiny networks usually have an insufficient number of channels to process the information.

### 3.3. Network Architecture

## 4. Experiments

> The core idea of ShuffleNet lies in pointwise group convolution and channel shuffle operation. In this subsection we evaluate them respectively.

### 4.1. Ablation Study

### 4.2. Comparison with Other Structure Units

### 4.3. Comparison with MobileNets and Other Frameworks

### 4.4. Generalization Ability

### 4.5. Actual Speedup Evaluation

----------------------------------------------------------------------------------------------------

## References

* Zhang, Xiangyu, et al. "Shufflenet: An extremely efficient convolutional neural network for mobile devices." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

## Further Reading

* [3] [Xception](https://zhuanlan.zhihu.com/p/556794897)
* [5] R-CNN
* [9] ResNet
* [10] Identity Mappings in Deep Residual Networks
* [12] MobileNetV1
* [13] SENet
* [14] SqueezeNet
* [21] AlexNet
* [24] FCN
* [28] Faster R-CNN
* [30] VGG
* [32] Inception-v4
* [33] Inception-v1/GoogLeNet
* [34] Inception-v3
* [40] ResNeXt
* [46] NASNet
