# SSD: Single Shot MultiBox Detector

## Default Boxes

* Similar to anchor boxes but apply to several feature maps of different resolutions.

* A combination of feature pyramid and anchor mechanism.

## Loss Function

## Hard Negative Mining

* After the matching step, most of the default boxes are negatives, especially when the number of possible default boxes is large.
* This introduces a significant imbalance between the positive and negative training examples.
* Instead of using all the negative examples, we sort them using the highest confidence loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most 3:1.
