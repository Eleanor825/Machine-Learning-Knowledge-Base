# [Notes][Vision][Detection] OverFeat

* url: https://arxiv.org/abs/1312.6229
* Title: OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
* Year: 21 Dec `2013`
* Authors: Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, Yann LeCun
* Institutions: [Courant Institute of Mathematical Sciences, New York University]
* Abstract: We present an integrated framework for using Convolutional Networks for classification, localization and detection. We show how a multiscale and sliding window approach can be efficiently implemented within a ConvNet. We also introduce a novel deep learning approach to localization by learning to predict object boundaries. Bounding boxes are then accumulated rather than suppressed in order to increase detection confidence. We show that different tasks can be learned simultaneously using a single shared network. This integrated framework is the winner of the localization task of the ImageNet Large Scale Visual Recognition Challenge 2013 (ILSVRC2013) and obtained very competitive results for the detection and classifications tasks. In post-competition work, we establish a new state of the art for the detection task. Finally, we release a feature extractor from our best model called OverFeat.

----------------------------------------------------------------------------------------------------

## 1 Introduction

> The main `advantage` of ConvNets for many such tasks is that the entire system is trained end to end, from raw pixels to ultimate categories, thereby alleviating the requirement to manually design a suitable feature extractor.

> The main `disadvantage` is their ravenous appetite for labeled training samples.

> The main point of this paper is to show that training a convolutional network to simultaneously classify, locate and detect objects in images can boost the classification accuracy and the detection and localization accuracy of all tasks.

> The first idea in addressing this is to apply a ConvNet at multiple locations in the image, in a sliding window fashion, and over multiple scales.

> The second idea is to train the system to not only produce a distribution over categories for each window, but also to produce a prediction of the location and size of the bounding box containing the object relative to the window.

> The third idea is to accumulate the evidence for each category at each location and size.

## 2 Vision Tasks

> In this paper, we explore three computer vision tasks in increasing order of difficulty: (i) classification, (ii) localization, and (iii) detection. Each task is a sub-task of the next. While all tasks are addressed using a single framework and a shared feature learning base, we will describe them separately in the following sections.

## 3 Classification

### 3.1 Model Design and Training

### 3.2 Feature Extractor

### 3.3 Multi-Scale Classification

> Instead, we explore the entire image by densely running the network at each location and at multiple scales.

----------------------------------------------------------------------------------------------------

## References

* Sermanet, Pierre, et al. "Overfeat: Integrated recognition, localization and detection using convolutional networks." *arXiv preprint arXiv:1312.6229* (2013).

## Further Reading

* [15] AlexNet
