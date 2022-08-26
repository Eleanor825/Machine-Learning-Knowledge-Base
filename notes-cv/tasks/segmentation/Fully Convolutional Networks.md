# [Notes][Vision][Segmentation] Fully Convolutional Networks

* url: https://arxiv.org/abs/1411.4038
* Title: Fully Convolutional Networks for Semantic Segmentation
* Year: 14 Nov `2014`
* Authors: Jonathan Long, Evan Shelhamer, Trevor Darrell
* Abstract: Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build "fully convolutional" networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet, the VGG net, and GoogLeNet) into fully convolutional networks and transfer their learned representations by fine-tuning to the segmentation task. We then define a novel architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional network achieves state-of-the-art segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes one third of a second for a typical image.

----------------------------------------------------------------------------------------------------

## 1. Introduction

## 2. Related work

## 3. Fully convolutional networks

Notations:
* Let $H, W \in \mathbb{Z}_{++}$ denote the height and width of the input feature map to the layer.
* Let $x_{i,j} \in \mathbb{R}^{c_{1}}$ denote the data vector at location $(i, j)$ of the input feature map to the layer.
* Let $y_{i,j} \in \mathbb{R}^{c_{2}}$ denote the data vector at location $(i, j)$ of the output feature map of the layer.
* Let $k \in \mathbb{Z}_{++}$ denote the kernel size.
* Let $s \in \mathbb{Z}_{++}$ denote the strides.
* Let $f_{ks}: \mathbb{R}^{H \times W \times c_{1}} \to \mathbb{R}^{H \times W \times c_{2}}$ denote the layer.

Then
$$y_{i,j} = f_{ks}(\{x_{s \cdot i+\Delta{i}, s \cdot j + \Delta{j}}: 0 \leq \Delta{i}, \Delta{j} \leq k\}).$$

> This functional form is maintained under composition, with kernel size and stride obeying the transformation rule

$$f_{ks} \circ g_{k's'} = (f \circ g)_{k'+(k-1)s', ss'}.$$

> An FCN naturally operates on an input of any size, and produces an output of corresponding (possibly resampled) spatial dimensions.

### 3.1. Adapting classifiers for dense prediction

> Typical recognition nets, including LeNet [21], AlexNet [19], and its deeper successors [31, 32], ostensibly take fixed-sized inputs and produce nonspatial outputs. The fully connected layers of these nets have fixed dimensions and throw away spatial coordinates.

> However, these fully connected layers can also be viewed as convolutions with kernels that cover their entire input regions. Doing so casts them into fully convolutional networks that take input of any size and output classification maps.

> The spatial output maps of these convolutionalized models make them a natural choice for dense problems like semantic segmentation. With ground truth available at every output cell, both the forward and backward passes are straightforward, and both take advantage of the inherent computational efficiency (and aggressive optimization) of convolution.

### 3.2. Shift-and-stitch is filter rarefaction

### 3.3. Upsampling is backwards strided convolution

### 3.4. Patchwise training is loss sampling

## 4. Segmentation Architecture

### 4.1. From classifier to dense FCN

### 4.2. Combining what and where

### 4.3. Experimental framework

## 5. Results

## 6. Conclusion

----------------------------------------------------------------------------------------------------

## References

* Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2015.

## Further Reading

* [17] Spatial Pyramid Pooling (SPP)
* [19] AlexNet
* [21] LeNet
* [29] OverFeat
* [31] VGG
* [32] Inception-v1/GoogLeNet
