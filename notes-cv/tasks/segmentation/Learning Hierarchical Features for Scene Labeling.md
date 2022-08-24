# [Notes][Vision][Segmentation] Learning Hierarchical Features for Scene Labeling

* url: https://ieeexplore.ieee.org/document/6338939
* Title: Learning Hierarchical Features for Scene Labeling
* Year: `2013`
* Authors: Clement Farabet; Camille Couprie; Laurent Najman; Yann LeCun
* Abstract: Scene labeling consists of labeling each pixel in an image with the category of the object it belongs to. We propose a method that uses a multiscale convolutional network trained from raw pixels to extract dense feature vectors that encode regions of multiple sizes centered on each pixel. The method alleviates the need for engineered features, and produces a powerful representation that captures texture, shape, and contextual information. We report results using multiple postprocessing methods to produce the final labeling. Among those, we propose a technique to automatically retrieve, from a pool of segmentation components, an optimal set of components that best explain the scene; these components are arbitrary, for example, they can be taken from a segmentation tree or from any family of oversegmentations. The system yields record accuracies on the SIFT Flow dataset (33 classes) and the Barcelona dataset (170 classes) and near-record accuracy on Stanford background dataset (eight classes), while being an order of magnitude faster than competing approaches, producing a 320×240 image labeling in less than a second, including feature extraction.

----------------------------------------------------------------------------------------------------

## 1 INTRODUCTION

> One challenge of scene parsing is that it combines the traditional problems of detection, segmentation, and multilabel recognition in a single process.

> There are two questions of primary importance in the context of scene parsing: how to produce good internal representations of the visual information, and how to use contextual information to ensure the self-consistency of the interpretation.

> Unfortunately, labeling each pixel by looking at a small region around it is difficult. The category of a pixel may depend on relatively short-range information (e.g., the presence of a human face generally indicates the presence of a human body nearby), but may also depend on long-range information. For example, identifying a gray pixel as belonging to a road, a sidewalk, a gray car, a concrete building, or a cloudy sky requires a wide contextual window that shows enough of the surroundings to make an informed decision.

> To address this problem, we propose using a multiscale ConvNet, which can take into account large input windows while keeping the number of free parameters to a minimum.

### 1.1 Multiscale, Convolutional Representation

### 1.2 Graph-Based Classification

#### 1.2.1 Superpixels

#### 1.2.2 CRF over Superpixels

#### 1.2.3 Multilevel Cut with Class Purity Criterion

## 2 RELATED WORK

## 3 MULTISCALE FEATURE EXTRACTION FOR SCENE PARSING

> The model proposed in this paper, depicted in Fig. 1, relies on two complementary image representations.

> In the first representation, an image patch is seen as a point in $\mathbb{R}^{P}$, and we seek to find a transform $f: \mathbb{R}^{P} \to \mathbb{R}^{Q}$ that maps each patch into $\mathbb{R}^{Q}$, a space where it can be classified linearly.

> This first representation typically suffers from two main problems when using a classical ConvNet where the image is divided following a grid pattern:
> 1. The window considered rarely contains an object that is properly centered and scaled, and therefore offers a poor observation basis to predict the class of the underlying object;
> 2. integrating a large context involves increasing the grid size and therefore the dimensionality $P$ of the input; given a finite amount of training data, it is then necessary to enforce some invariance in the function $f$ itself. This is usually achieved by using pooling/subsampling layers, which in turn degrades the ability of the model to precisely locate and delineate objects.

> In this paper, f is implemented by a multiscale ConvNet, which allows integrating large contexts (as large as the complete scene) into local decisions, while still remaining manageable in terms of parameters/dimensionality. This multiscale model in which weights are shared across scales allows the model to capture long-range interactions without the penalty of extra parameters to train.

> In the second representation, the image is seen as an edge-weighted graph on which one or several oversegmentations can be constructed.

### 3.1 Scale-Invariant, Scene-Level Feature Extraction

> Good internal representations are hierarchical. In vision, pixels are assembled into edglets, edglets into motifs, motifs into parts, parts into objects, and objects into scenes. This suggests that recognition architectures for vision (and for other modalities such as audio and natural language) should have multiple trainable stages stacked on top of each other, one for each level in the feature hierarchy. ConvNets provide a simple framework to learn such hierarchies of features.

Notations:
* Let $I$ denote the input image.
* Let $N \in \mathbb{Z}_{++}$ denote the number of levels in the pyramid.
* Let $g_{s}: \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{C \times H \times W}$ the scaling/normalization function at level $s$ of the pyramid, for $s \in \{1, ..., N\}$.
* Define $X_{s} \in \mathbb{R}^{C \times H \times W}$ by $X_{s} := g_{s}(I)$.

### 3.2 Learning Discriminative Scale-Invariant Features

Notations:
* Let $k \in \mathbb{Z}_{++}$ denote the number of classes.
* Let $\hat{c}_{i} \in \mathbb{R}^{k}$ denote the normalized prediction vector from the linear classifier for pixel $i$.

## 4 SCENE LABELING STRATEGIES

Notations:
* Define $l_{i} := \underset{a \in \{1, ..., k\}}{\operatorname{argmax}}\ \hat{c}_{i, a}$ for each pixel $i$.

### 4.1 Superpixels

### 4.2 Conditional Random Fields

### 4.3 Parameter-Free Multilevel Parsing



----------------------------------------------------------------------------------------------------

## References

* Farabet, Clement, et al. "Learning hierarchical features for scene labeling." *IEEE transactions on pattern analysis and machine intelligence* 35.8 (2012): 1915-1929.

## Further Reading

