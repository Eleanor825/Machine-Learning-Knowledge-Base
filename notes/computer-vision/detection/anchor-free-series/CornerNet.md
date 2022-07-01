<link rel="stylesheet" href="../../../style.css">

# [CornerNet](https://arxiv.org/abs/1808.01244)

* Title: CornerNet: Detecting Objects as Paired Keypoints
* Year: 03 Aug `2018`
* Author: Hei Law
* We propose CornerNet, a new approach to object detection where we detect an object bounding box as a pair of keypoints, the top-left corner and the bottom-right corner, using a single convolution neural network. By detecting objects as paired keypoints, we eliminate the need for designing a set of anchor boxes commonly used in prior single-stage detectors. In addition to our novel formulation, we introduce corner pooling, a new type of pooling layer that helps the network better localize corners. Experiments show that CornerNet achieves a 42.2% AP on MS COCO, outperforming all existing one-stage detectors.

----------------------------------------------------------------------------------------------------

## Introduction

> The use of anchor boxes has two drawbacks. First, we typically need a very large set of anchor boxes, e.g. more than 40k in DSSD (Fu et al., 2017) and more than 100k in RetinaNet (Lin et al., 2017). This is because the detector is trained to classify whether each anchor box sufficiently overlaps with a ground truth box, and a large number of anchor boxes is needed to ensure sufficient overlap with most ground truth boxes. As a result, only a tiny fraction of anchor boxes will overlap with ground truth; this creates a huge imbalance between positive and negative anchor boxes and slows down training (Lin et al., 2017).

> Second, the use of anchor boxes introduces many hyperparameters and design choices. These include how many boxes, what sizes, and what aspect ratios. Such choices have largely been made via ad-hoc heuristics, and can become even more complicated when combined with multiscale architectures where a single network makes separate predictions at multiple resolutions, with each scale using different features and its own set of anchor boxes (Liu et al., 2016; Fu et al., 2017; Lin et al., 2017).

> We detect an object as a pair of keypoints — the top-left corner and bottom-right corner of the bounding box. We use a single convolutional network to predict a heatmap for the top-left corners of all instances of the same object category, a heatmap for all bottom-right corners, and an embedding vector for each detected corner. The embeddings serve to group a pair of corners that belong to the same object—the network is trained to predict similar embeddings for them.

## Related Works

## CornerNet

### Overview

> In CornerNet, we detect an object as a pair of keypoints — the top-left corner and bottom-right corner of the bounding box. A convolutional network predicts two sets of `heatmaps` to represent the locations of corners of different object categories, one set for the top-left corners and the other for the bottom-right corners. The network also predicts an `embedding` vector for each detected corner such that the distance between the embeddings of two corners from the same object is small. To produce tighter bounding boxes, the network also predicts `offsets` to slightly adjust the locations of the corners. With the predicted heatmaps, embeddings and offsets, we apply a simple post-processing algorithm to obtain the final bounding boxes.

> We use the hourglass network as the `backbone network` of CornerNet. The hourglass network is followed by two `prediction modules`. One module is for the top-left corners, while the other one is for the bottom-right corners. Each module has its own `corner pooling module` to pool features from the hourglass network before predicting the heatmaps, embeddings and offsets.

### Detecting Corners

$$\mathcal{L}_{\text{det}} := -\frac{1}{N}\sum_{c=1}^{C}\sum_{i=1}^{H}\sum_{j=1}^{W}
\begin{cases}
(1-p_{cij})^{\alpha}\log(p_{cij}), & \text{ if } y_{cij} = 1, \\
(1-y_{cij})^{\beta}(p_{cij})^{\alpha}\log(1-p_{cij}), & \text{ otherwise. }
\end{cases}$$
With the Gaussian bumps encoded in $y_{cij}$, the $(1-y_{cij})$ term reduces the penalty around the ground truth locations.

$$\mathcal{L}_{\text{off}} := \frac{1}{N}\sum_{k=1}^{N}\text{SmoothL1Loss}(o_{k},\hat{o}_{k}).$$

### Grouping Corners

Let $e_{t_{k}}$ denote the embedding of the top-left corner of the bounding box around object $k$.

Let $e_{b_{k}}$ denote the embedding of the bottom-right corner of the bounding box around object $k$.

Let $e_{k} := \frac{1}{2}(e_{t_{k}}+e_{b_{k}})$.

Let $\Delta := 1$.

$$\mathcal{L}_{\text{pull}} = \frac{1}{N}\sum_{k=1}^{N}\bigg[(e_{t_{k}}-e_{k})^{2} + (e_{b_{k}}-e_{k})^{2}\bigg].$$

$$\mathcal{L}_{\text{push}} = \frac{1}{N(N-1)}\sum_{k=1}^{N}\sum_{j \neq k}\max(0, \Delta-|e_{j}-e_{k}|).$$

### Corner Pooling

