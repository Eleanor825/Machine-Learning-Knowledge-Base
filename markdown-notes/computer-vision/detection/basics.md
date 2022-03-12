# Object Detection Basics

## Anchor Mechanism

* A replacement of pyramids of images and pyramids of features.
  Anchor mechanism, pyramids of images, and pyramids of features are three different approaches of solving the same problem.
* Generate multiple anchors for each cell.
* Fixed the ground truth box overlapping problem.


## Feature Pyramid Network (FPN)
* The higher resolution features is upsampled spatially coarser, but semantically stronger, feature maps from higher pyramid levels.
* Specifically, the feature maps from bottom-up pathway undergoes 1×1 convolutions to reduce the channel dimensions.
* Lateral Connection:
    * Each lateral connection merges feature maps of the same spatial size from the bottom-up pathway and the top-down pathway.
    * The feature maps from the bottom-up pathway and the top-down pathway are merged by element-wise addition.
