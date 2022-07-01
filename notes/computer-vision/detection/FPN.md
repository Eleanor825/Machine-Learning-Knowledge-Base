<link rel="stylesheet" href="../../style.css">

# [FPN](https://arxiv.org/abs/1612.03144)

* Title: Feature Pyramid Networks for Object Detection
* Year: 09 Dec `2016`
* Author: Tsung-Yi Lin
* Abstract: Feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive. In this paper, we exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A top-down architecture with lateral connections is developed for building high-level semantic feature maps at all scales. This architecture, called a Feature Pyramid Network (FPN), shows significant improvement as a generic feature extractor in several applications. Using FPN in a basic Faster R-CNN system, our method achieves state-of-the-art single-model results on the COCO detection benchmark without bells and whistles, surpassing all existing single-model entries including those from the COCO 2016 challenge winners. In addition, our method can run at 5 FPS on a GPU and thus is a practical and accurate solution to multi-scale object detection. Code will be made publicly available.

----------------------------------------------------------------------------------------------------

> The goal of this paper is to naturally leverage the pyramidal shape of a ConvNet's `feature hierarchy` while creating a feature pyramid that has `strong semantics at all scales`. To achieve this goal, we rely on an architecture that combines `low-resolution`, `semantically strong` features with `high-resolution`, `semantically weak` features via a top-down pathway and lateral connections. The result is a feature pyramid that has rich semantics at all levels and is built quickly from a single input image scale. In other words, we show how to create `in-network feature pyramids` that can be used to `replace featurized image pyramids` without sacrificing representational power, speed, or memory.

## Model Architecture

* Bottom-up Pathway

> The bottom-up pathway is the feed-forward computation of the `backbone` ConvNet, which computes a `feature hierarchy` consisting of feature maps at several scales with a scaling step of 2. There are often many layers producing output maps of the same size and we say these layers are in the same `network stage`. For our feature pyramid, we define one pyramid level for each stage. We choose the output of the last layer of each stage as our reference set of feature maps, which we will enrich to create our pyramid. This choice is natural since the deepest layer of each stage should have the strongest features.

* Top-down Pathway and Lateral Connections

> The top-down pathway hallucinates higher resolution features by upsamping spatially coarser, but semantically stronger, feature maps from higher pyramid levels. These features are then enhanced with features from the bottom-up pathway via lateral connections. Each lateral connection merges feature maps of the same spatial size from the bottom-up pathway and the top-down pathway. The bottom-up feature map is of lower-level semantics, but its activations are more accurately localized as it was subsampled fewer times.

* The higher resolution features is upsampled spatially coarser, but semantically stronger, feature maps from higher pyramid levels.
* Specifically, the feature maps from bottom-up pathway undergoes 1×1 convolutions to reduce the channel dimensions.
* Lateral Connection:
    * Each lateral connection merges feature maps of the same spatial size from the bottom-up pathway and the top-down pathway.
    * The feature maps from the bottom-up pathway and the top-down pathway are merged by element-wise addition.
