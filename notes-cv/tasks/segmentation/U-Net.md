# [Notes][Vision][Segmentation] U-Net

* url: https://arxiv.org/abs/1505.04597
* Title: U-Net: Convolutional Networks for Biomedical Image Segmentation
* Year: 18 May `2015`
* Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
* Institutions: [Computer Science Department and BIOSS Centre for Biological Signalling Studies, University of Freiburg, Germany]
* Abstract: There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at [this http URL](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net).

----------------------------------------------------------------------------------------------------

## 1 Introduction

> Ciresan et al. [1] trained a network in a sliding-window setup to predict the class label of each pixel by providing a local region (patch) around that pixel as input.
> 1. First, this network can localize.
> 2. Secondly, the training data in terms of patches is much larger than the number of training images.

> Obviously, the strategy in Ciresan et al. [1] has two drawbacks.
> 1. First, it is quite slow because the network must be run separately for each patch, and there is a lot of redundancy due to overlapping patches.
> 2. Secondly, there is a trade-off between localization accuracy and the use of context. Larger patches require more max-pooling layers that reduce the localization accuracy, while small patches allow the network to see only little context.

> More recent approaches [11, 4] proposed a classifier output that takes into account the features from multiple layers. Good localization and the use of context are possible at the same time.

Model Architecture

> In this paper, we build upon a more elegant architecture, the so-called "fully convolutional network" [9].

> One important modification in our architecture is that in the upsampling part we have also a large number of feature channels, which allow the network to propagate context information to higher resolution layers. As a consequence, the expansive path is more or less symmetric to the contracting path, and yields a u-shaped architecture.

> This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy (see Figure 2). To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image.

Data Augmentation

> As for our tasks there is very little training data available, we use excessive data augmentation by applying `elastic deformations` to the available training images. This allows the network to learn invariance to such deformations, without the need to see these transformations in the annotated image corpus. This is particularly important in biomedical segmentation, since deformation used to be the most common variation in tissue and realistic deformations can be simulated efficiently.

Loss Function

> Another challenge in many cell segmentation tasks is the separation of touching objects of the same class; see Figure 3. To this end, we propose the use of a weighted loss, where the separating background labels between touching cells obtain a large weight in the loss function.

## 2 Network Architecture

> It consists of a `contracting` path (left side) and an `expansive` path (right side).

Contracting Path

> The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels.

Expansive Path

> Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution ("up-convolution") that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution.

Final Layer

> At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes.

## 3 Training

> Due to the unpadded convolutions, the output image is smaller than the input by a constant border width.

> The energy function is computed by a pixel-wise soft-max over the final feature map combined with the cross entropy loss function.

Weight Map

> We pre-compute the weight map for each ground truth segmentation to compensate the different frequency of pixels from a certain class in the training data set, and to force the network to learn the small separation borders that we introduce between touching cells (See Figure 3c and d).

Network Initialization

### 3.1 Data Augmentation

> Data augmentation is essential to teach the network the desired invariance and robustness properties, when only few training samples are available.

> We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid.

> Drop-out layers at the end of the contracting path perform further implicit data augmentation.

## 4 Experiments

## 5 Conclusion

----------------------------------------------------------------------------------------------------

## References

* Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." *International Conference on Medical image computing and computer-assisted intervention*. Springer, Cham, 2015.

## Further Reading

* [3] R-CNN
* [7] AlexNet
* [9] FCN
* [12] VGG
