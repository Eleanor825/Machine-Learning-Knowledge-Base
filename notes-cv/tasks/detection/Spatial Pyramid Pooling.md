# [Notes][Vision][Detection] Spatial Pyramid Pooling (SPP)

* url: https://arxiv.org/abs/1406.4729
* Title: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
* Year: 18 Jun `2014`
* Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
* Institutions: [Microsoft Research, Beijing, China], [Xi'an Jiaotong University, Xi'an, China], [University of Science and Technology of China, Hefei, China]
* Abstract: Existing deep convolutional neural networks (CNNs) require a fixed-size (e.g., 224x224) input image. This requirement is "artificial" and may reduce the recognition accuracy for the images or sub-images of an arbitrary size/scale. In this work, we equip the networks with another pooling strategy, "spatial pyramid pooling", to eliminate the above requirement. The new network structure, called SPP-net, can generate a fixed-length representation regardless of image size/scale. Pyramid pooling is also robust to object deformations. With these advantages, SPP-net should in general improve all CNN-based image classification methods. On the ImageNet 2012 dataset, we demonstrate that SPP-net boosts the accuracy of a variety of CNN architectures despite their different designs. On the Pascal VOC 2007 and Caltech101 datasets, SPP-net achieves state-of-the-art classification results using a single full-image representation and no fine-tuning. The power of SPP-net is also significant in object detection. Using SPP-net, we compute the feature maps from the entire image only once, and then pool features in arbitrary regions (sub-images) to generate fixed-length representations for training the detectors. This method avoids repeatedly computing the convolutional features. In processing test images, our method is 24-102x faster than the R-CNN method, while achieving better or comparable accuracy on Pascal VOC 2007. In ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014, our methods rank #2 in object detection and #3 in image classification among all 38 teams. This manuscript also introduces the improvement made for this competition.

----------------------------------------------------------------------------------------------------

### 2.2 The Spatial Pyramid Pooling Layer

> The convolutional layers accept arbitrary input sizes, but they produce outputs of variable sizes.

> Spatial pyramid pooling [14, 15] improves BoW in that it can maintain spatial information by pooling in local spatial bins. These spatial bins have sizes proportional to the image size, so the number of bins is fixed regardless of the image size.

> This is in contrast to the sliding window pooling of the previous deep networks [3], where the number of sliding windows depends on the input size.

> To adopt the deep network for images of arbitrary sizes, we replace the last pooling layer (e.g., pool5, after the last convolutional layer) with a spatial pyramid pooling layer.

> The fixed-dimensional vectors are the input to the fully-connected layer.

> With spatial pyramid pooling, the input image can be of any sizes. This not only allows arbitrary aspect ratios, but also allows arbitrary scales. We can resize the input image to any scale (e.g., min(w,h)=180, 224, ...) and apply the same deep network.

> When the input image is at different scales, the network (with the same filter sizes) will extract features at different scales.

> Interestingly, the coarsest pyramid level has a single bin that covers the entire image. This is in fact a “global pooling” operation, which is also investigated in several concurrent works.

### 2.3 Training the Network

----------------------------------------------------------------------------------------------------

## References

* He, Kaiming, et al. "Spatial pyramid pooling in deep convolutional networks for visual recognition." *IEEE transactions on pattern analysis and machine intelligence* 37.9 (2015): 1904-1916.

## Further Reading

* [5] OverFeat
* [31] Network In Network (NIN)
* [32] [InceptionNetV1](https://zhuanlan.zhihu.com/p/564141144)
