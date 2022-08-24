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



----------------------------------------------------------------------------------------------------

## References

* Farabet, Clement, et al. "Learning hierarchical features for scene labeling." *IEEE transactions on pattern analysis and machine intelligence* 35.8 (2012): 1915-1929.

## Further Reading

