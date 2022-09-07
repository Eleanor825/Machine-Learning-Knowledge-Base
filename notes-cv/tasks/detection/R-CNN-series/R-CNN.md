# [Notes][Vision][Detection] R-CNN

* url: https://arxiv.org/abs/1311.2524
* Title: Rich feature hierarchies for accurate object detection and semantic segmentation
* Year: 11 Nov `2013`
* Authors: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik
* Institutions: [UC Berkeley]
* Abstract: Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012---achieving a mAP of 53.3%. Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also compare R-CNN to OverFeat, a recently proposed sliding-window detector based on a similar CNN architecture. We find that R-CNN outperforms OverFeat by a large margin on the 200-class ILSVRC2013 detection dataset. Source code for the complete system is available at [this http URL](http://www.cs.berkeley.edu/~rbg/rcnn).

----------------------------------------------------------------------------------------------------

## 1. Introduction

> The last decade of progress on various visual recognition tasks has been based considerably on the use of SIFT [29] and HOG [7].

> CNNs saw heavy use in the 1990s (e.g., [27]), but then fell out of fashion with the rise of support vector machines. In 2012, Krizhevsky et al. [25] rekindled interest in CNNs by showing substantially higher image classification accuracy on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) [9, 10].

The first challenge: localization

> One approach frames localization as a regression problem. However, work from Szegedy et al. [38], concurrent with our own, indicates that this strategy may not fare well in practice (they report a mAP of 30.5% on VOC 2007 compared to the 58.5% achieved by our method).

> An alternative is to build a sliding-window detector. CNNs have been used in this way for at least two decades, typically on constrained object categories, such as faces [32, 40] and pedestrians [35].

> Instead, we solve the CNN localization problem by operating within the “recognition using regions” paradigm [21], which has been successful for both object detection [39] and semantic segmentation [5].

The second challenge: labeled data is scarce

> The conventional solution to this problem is to use unsupervised pre-training, followed by supervised fine-tuning (e.g., [35]).

> The second principle contribution of this paper is to show that supervised pre-training on a large auxiliary dataset (ILSVRC), followed by domain-specific fine-tuning on a small dataset (PASCAL), is an effective paradigm for learning high-capacity CNNs when data is scarce.

## 2. Object detection with R-CNN

> Our object detection system consists of three modules.
> 1. The first generates category-independent region proposals. These proposals define the set of candidate detections available to our detector.
> 2. The second module is a large convolutional neural network that extracts a fixed-length feature vector from each region.
> 3. The third module is a set of class-specific linear SVMs.

### 2.1. Module design

**Region proposals**

**Feature extraction**

> We extract a 4096-dimensional feature vector from each region proposal using the Caffe [24] implementation of the CNN described by Krizhevsky et al. [25].

### 2.2. Test-time detection

> 1. run selective search on the test image to extract around 2000 region proposals
> 2. warp each proposal
> 3. forward propagate it through the CNN in order to compute features
> 4. for each class, we score each extracted feature vector using the SVM trained for that class
> 5. apply a greedy non-maximum suppression (for each class independently) that rejects a region if it has an intersection-over-union (IoU) overlap with a higher scoring selected region larger than a learned threshold.

**Run-time analysis**

> Two properties make detection efficient.
> 1. all CNN parameters are shared across all categories.
> 2. the feature vectors computed by the CNN are low-dimensional when compared to other common approaches, such as spatial pyramids with bag-of-visual-word encodings.

### 2.3. Training

**Supervised pre-training**

> We discriminatively pre-trained the CNN on a large auxiliary dataset (ILSVRC2012 classification) using image-level annotations only (bounding-box labels are not available for this data).

**Domain-specific fine-tuning**

> To adapt our CNN to the new task (detection) and the new domain (warped proposal windows), we continue stochastic gradient descent (SGD) training of the CNN parameters using only warped region proposals.

> Aside from replacing the CNN’s ImageNet-specific 1000-way classification layer with a randomly initialized (N+1)-way classification layer (where N is the number of object classes, plus 1 for background), the CNN architecture is unchanged.

Data Sampling

> We treat all region proposals with $\geq 0.5$ IoU overlap with a ground-truth box as positives for that box’s class and the rest as negatives.

> In each SGD iteration, we uniformly sample 32 positive windows (over all classes) and 96 background windows to construct a mini-batch of size 128. We bias the sampling towards positive windows because they are extremely rare compared to background.

**Object category classifiers**

Setting a threshold

> It’s clear that an image region tightly enclosing a car should be a positive example. Similarly, it’s clear that a background region, which has nothing to do with cars, should be a negative example. Less clear is how to label a region that partially overlaps a car. We resolve this issue with an IoU overlap threshold, below which regions are defined as negatives.

How to choose the threshold

> The overlap threshold, 0.3, was selected by a grid search over {0,0.1,...,0.5} on a validation set. We found that selecting this threshold carefully is important.

Hard Negative Mining

> Since the training data is too large to fit in memory, we adopt the standard hard negative mining method [17, 37]. Hard negative mining converges quickly and in practice mAP stops increasing after only a single pass over all images.

### 2.4. Results on PASCAL VOC 2010-12

### 2.5. Results on ILSVRC2013 detection

## 3. Visualization, ablation, and modes of error

## 4. The ILSVRC2013 detection dataset

## 5. Semantic segmentation

## 6. Conclusion

----------------------------------------------------------------------------------------------------

## References

* Girshick, Ross, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2014.

## Further Reading

* [12] AlexNet
* [34] OverFeat
