# [Notes][Vision][Segmentation] DeepLabV1

* url: https://arxiv.org/abs/1412.7062
* Title: Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
* Year: 22 Dec `2014`
* Authors: Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
* Institutions: Univ. of California, Google Inc., CentraleSupelec and INRIA
* Abstract: Deep Convolutional Neural Networks (DCNNs) have recently shown state of the art performance in high level vision tasks, such as image classification and object detection. This work brings together methods from DCNNs and probabilistic graphical models for addressing the task of pixel-level classification (also called "semantic image segmentation"). We show that responses at the final layer of DCNNs are not sufficiently localized for accurate object segmentation. This is due to the very invariance properties that make DCNNs good for high level tasks. We overcome this poor localization property of deep networks by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF). Qualitatively, our "DeepLab" system is able to localize segment boundaries at a level of accuracy which is beyond previous methods. Quantitatively, our method sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 71.6% IOU accuracy in the test set. We show how these results can be obtained efficiently: Careful network re-purposing and a novel application of the 'hole' algorithm from the wavelet community allow dense computation of neural net responses at 8 frames per second on a modern GPU.

----------------------------------------------------------------------------------------------------

## 1 INTRODUCTION

## 2 RELATED WORK

## 3 CONVOLUTIONAL NEURAL NETWORKS FOR DENSE IMAGE LABELING

### 3.1 EFFICIENT DENSE SLIDING WINDOW FEATURE EXTRACTION WITH THE HOLE ALGORITHM



----------------------------------------------------------------------------------------------------

## References

* Chen, Liang-Chieh, et al. "Semantic image segmentation with deep convolutional nets and fully connected crfs." *arXiv preprint arXiv:1412.7062* (2014).

## Further Reading

* [Simonyan & Zisserman, 2014] VGG
* [Sermanet et al., 2013] OverFeat
* [Szegedy et al., 2014] [InceptionNetV1/GoogLeNet](https://zhuanlan.zhihu.com/p/564141144)
* [Girshick et al., 2014] R-CNN
* [Long et al., 2014] [Fully Convolutional Networks (FCN)](https://zhuanlan.zhihu.com/p/561031110)
