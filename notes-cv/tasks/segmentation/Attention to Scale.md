# [Notes][Vision][Segmentation] Attention to Scale

* url: https://arxiv.org/abs/1511.03339
* Title: Attention to Scale: Scale-aware Semantic Image Segmentation
* Year: 10 Nov `2015`
* Authors: Liang-Chieh Chen, Yi Yang, Jiang Wang, Wei Xu, Alan L. Yuille
* Institutions: [Baidu USA]
* Abstract: Incorporating multi-scale features in fully convolutional neural networks (FCNs) has been a key element to achieving state-of-the-art performance on semantic image segmentation. One common way to extract multi-scale features is to feed multiple resized input images to a shared deep network and then merge the resulting features for pixelwise classification. In this work, we propose an attention mechanism that learns to softly weight the multi-scale features at each pixel location. We adapt a state-of-the-art semantic image segmentation model, which we jointly train with multi-scale input images and the attention model. The proposed attention model not only outperforms average- and max-pooling, but allows us to diagnostically visualize the importance of features at different positions and scales. Moreover, we show that adding extra supervision to the output at each scale is essential to achieving excellent performance when merging multi-scale features. We demonstrate the effectiveness of our model with extensive experiments on three challenging datasets, including PASCAL-Person-Part, PASCAL VOC 2012 and a subset of MS-COCO 2014.

----------------------------------------------------------------------------------------------------

## 1. Introduction

## 2. Related Work

## 3. Model

### 3.1. Review of DeepLab



----------------------------------------------------------------------------------------------------

## References

* Chen, Liang-Chieh, et al. "Attention to scale: Scale-aware semantic image segmentation." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

## Further Reading

* [24] R-CNN
* [28] Spatial Pyramid Pooling (SPP)
* [31] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [38] [Fully Convolutional Networks (FCN)](https://zhuanlan.zhihu.com/p/561031110)
* [47] OverFeat
* [49] [VGGNet](https://zhuanlan.zhihu.com/p/563314926)
* [50] [InceptionNetV1](https://zhuanlan.zhihu.com/p/564141144)
