## CNN Architectures

* AlexNet
* VGGNet
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

## Detection

* R-CNN Series
* YOLO Series
* RetinaNet

## Segmentation

## Questions

> What are some regularization techniques?

Answer:
1. (2012, Dropout) Dropout.
2. (2012, AlexNet) Weight decay.
2. Auxiliary classifiers (Inception-v3, section 4)
3. Batch Normalization (a conjecture, Inception-v3, section 4)
4. Label smoothing regularization (LSR) (Inception-v3, section 7)

> What are some ways to reduce overfitting?

Answer:
1. (2012, AlexNet) The easiest and most common method to reduce overfitting on image data is to artificially enlarge the dataset using label-preserving transformations (e.g., [25, 4, 5]).
2. (2012, AlexNet) Dropout.

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
