# [Notes][Vision][Segemntation] Hypercolumns

* url: https://arxiv.org/abs/1411.5752
* Title: Hypercolumns for Object Segmentation and Fine-grained Localization
* Year: 21 Nov `2014`
* Authors: Bharath Hariharan, Pablo Arbeláez, Ross Girshick, Jitendra Malik
* Institutions: [University of California, Berkeley], [Universidad de los Andes, Colombia], [Microsoft Research, Redmond]
* Abstract: Recognition algorithms based on convolutional networks (CNNs) typically use the output of the last layer as feature representation. However, the information in this layer may be too coarse to allow precise localization. On the contrary, earlier layers may be precise in localization but will not capture semantics. To get the best of both worlds, we define the hypercolumn at a pixel as the vector of activations of all CNN units above that pixel. Using hypercolumns as pixel descriptors, we show results on three fine-grained localization tasks: simultaneous detection and segmentation[22], where we improve state-of-the-art from 49.7[22] mean AP^r to 60.0, keypoint localization, where we get a 3.3 point boost over[20] and part labeling, where we show a 6.6 point gain over a strong baseline.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> Typically, recognition algorithms use the output of the last layer of the CNN. This makes sense when the task is assigning category labels to images or bounding boxes: the last layer is the most sensitive to category-level semantic information and the most invariant to “nuisance” variables such as pose, illumination, articulation, precise location and so on. However, when the task we are interested in is finer-grained, such as one of segmenting the detected object or estimating its pose, these nuisance variables are precisely what we are interested in. For such applications, the top layer is thus not the optimal representation.

> Our hypothesis is that the information of interest is distributed over all levels of the CNN and should be exploited in this way.

> We define the “hypercolumn” at a given input location as the outputs of all units above that location at all layers of the CNN, stacked into one vector.

## 2. Related work

**Combining features across multiple levels**

**Detection and segmentation**

## 3. Pixel classification using hypercolumns

**Problem setting**

> We assume an object detection system that gives us a set of detections. Each detection comes with a bounding box, a category label and a score (and sometimes an initial segmentation hypothesis).

> For every detection, we want to segment out the object, segment its parts or predict its keypoints.



----------------------------------------------------------------------------------------------------

## References

* Hariharan, Bharath, et al. "Hypercolumns for object segmentation and fine-grained localization." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2015.

## Further Reading

* [16] DPM
* [18] R-CNN
* [22] Simultaneous Detection and Segmentation (SDS)
* [28] AlexNet
* [36] VGG
