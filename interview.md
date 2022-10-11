# Interview

## CNN Architectures

* (2012) AlexNet
* (2014) VGGNet
* (2015) ResNet
* InceptionNets
* Xception
* MobileNets
* Flattened Networks
* Deconvolutions
* Dilated Convolutions
* Mixconv
* SqueezeNet
* EfficientNets
* ShuffleNets

## Transformer Architectures

* Transformer
* ViT
* DeiT
* Swin Transformer
* MobileViT
* PVT
* DETR
* Deformable DETR

## Detection

* R-CNN Series
* YOLO Series
* RetinaNet

## Segmentation

* Fully Convolutional Networks (FCN)

## Questions

> What are some ways to reduce overfitting?

Answer:
1. (2012, AlexNet) The easiest and most common method to reduce overfitting on image data is to artificially enlarge the dataset using label-preserving transformations (e.g., [25, 4, 5]).
1. (2012, AlexNet) Dropout.
1. Early stopping

> What are some regularization techniques?

Answer:
1. (2012, Dropout) Dropout.
1. (2012, AlexNet) Weight decay.
1. (2014, VGGNet) Implicit regularization: replace 7x7 kernels with stacks of 3x3 kernels, forcing them to be decomposable into 3x3 kernels.
1. (2015, InceptionNetV3, Section 4) Auxiliary classifiers
1. (2015, InceptionNetV3, Section 4) Batch normalization (a conjecture)
1. (2016, Layer Normalization, Section 1) Batch normalization: In addition to training time improvement, the stochasticity from the batch statistics serves as a regularizer during training.
1. (2015, InceptionNetV3, Section 7) Label smoothing regularization (LSR)

> What are some ways to augment data?

Answer:
1. (2012, AlexNet) Train on randomly extracted image patches.

> Explain Local Response Normalisation (LRN) (in AlexNet and VGGNet).

> Explain contrast normalization.

> Explain local contrast normalization (in [11] of AlexNet).

> Explain dropout.

> How to do classification on low resolution images?

> Explain uncertainty loss.

> How to deal with low quality dataset which contains a certain amount of wrong labels?
