# [Notes][Vision][Segmentation] Simultaneous Detection and Segmentation

* url: https://arxiv.org/abs/1407.1808
* Title: Simultaneous Detection and Segmentation
* Year: 07 Jul `2014`
* Authors: Bharath Hariharan, Pablo Arbeláez, Ross Girshick, Jitendra Malik
* Institutions: [University of California, Berkeley], [Universidad de los Andes, Colombia]
* Abstract: We aim to detect all instances of a category in an image and, for each instance, mark the pixels that belong to it. We call this task Simultaneous Detection and Segmentation (SDS). Unlike classical bounding box detection, SDS requires a segmentation and not just a box. Unlike classical semantic segmentation, we require individual object instances. We build on recent work that uses convolutional neural networks to classify category-independent region proposals (R-CNN [16]), introducing a novel architecture tailored for SDS. We then use category-specific, top- down figure-ground predictions to refine our bottom-up proposals. We show a 7 point boost (16% relative) over our baselines on SDS, a 5 point boost (10% relative) over state-of-the-art on semantic segmentation, and state-of-the-art performance in object detection. Finally, we provide diagnostic tools that unpack performance and provide directions for future work.

----------------------------------------------------------------------------------------------------

## 1 Introduction

> Object `recognition` comes in many flavors, two of the most popular being object `detection` and `semantic segmentation`.

> Starting with face detection, the task in object detection is to mark out bounding boxes around each object of a particular category in an image. In this task, a predicted bounding box is considered a true positive if it overlaps by more than 50% with a ground truth box, and different algorithms are compared based on their precision and recall.

> In contrast, semantic segmentation requires one to assign a category label to all pixels in an image.

> This task deals with "stuff" categories (such as grass, sky, road) and "thing" categories (such as cow, person, car) interchangeably. For things, this means that there is no notion of object instances. A typical semantic segmentation algorithm might accurately mark out the dog pixels in the image, but would provide no indication of how many dogs there are, or of the precise spatial extent of any one particular dog.

> Although often treated as separate problems, we believe the distinction between them is artificial. For the "thing" categories, we can think of a unified task: detect all instances of a category in an image and, for each instance, correctly mark the pixels that belong to it.

> Compared to the bounding boxes output by an object detection system or the pixel-level category labels output by a semantic segmentation system, this task demands a richer, and potentially more useful, output.

> The SDS algorithm we propose has the following steps:
> 1. Proposal generation
> 2. Feature extraction
> 3. Region classification
> 4. Region refinement

> Given an image, we expect the algorithm to produce a set of object hypotheses, where each hypothesis comes with a predicted segmentation and a score.

> One can argue that the 50% threshold is itself artificial.

> Thus the threshold at which we regard a detection as a true positive depends on the application. In general, we want algorithms that do well under a variety of thresholds.

> As the threshold varies, the PR curve traces out a PR surface. We can use the volume under this PR surface as a metric.

## 2 Related work



----------------------------------------------------------------------------------------------------

## References

* Hariharan, Bharath, et al. "Simultaneous detection and segmentation." *European conference on computer vision*. Springer, Cham, 2014.

## Further Reading

* [12] Learning Hierarchical Features
* [16] R-CNN
* [21] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [28] OverFeat
