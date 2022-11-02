#! https://zhuanlan.zhihu.com/p/579897223
# [Notes][Vision][Segmentation] ParseNet <!-- omit in toc -->

* urls: [[abs](https://arxiv.org/abs/1506.04579)]
    [[pdf](https://arxiv.org/pdf/1506.04579.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1506.04579/)]
* Title: ParseNet: Looking Wider to See Better
* Year: 15 Jun `2015`
* Authors: Wei Liu, Andrew Rabinovich, Alexander C. Berg
* Abstract: We present a technique for adding global context to deep convolutional networks for semantic segmentation. The approach is simple, using the average feature for a layer to augment the features at each location. In addition, we study several idiosyncrasies of training, significantly increasing the performance of baseline networks (e.g. from FCN). When we add our proposed global feature, and a technique for learning normalization parameters, accuracy increases consistently even over our improved versions of the baselines. Our proposed approach, ParseNet, achieves state-of-the-art performance on SiftFlow and PASCAL-Context with small additional computational cost over baselines, and near current state-of-the-art performance on PASCAL VOC 2012 semantic segmentation with a simple approach. Code is available at this https URL .

## Table of Contents <!-- omit in toc -->

- [Summary of Main Contributions](#summary-of-main-contributions)
- [1 INTRODUCTION](#1-introduction)
- [2 RELATED WORK](#2-related-work)
- [3 PARSENET](#3-parsenet)
  - [3.1 GLOBAL CONTEXT](#31-global-context)
  - [3.2 EARLY FUSION AND LATE FUSION](#32-early-fusion-and-late-fusion)
  - [3.3 $L_2$ NORMALIZATION LAYER](#33-l_2-normalization-layer)
- [4 EXPERIMENTS](#4-experiments)
- [5 CONCLUSION](#5-conclusion)
- [References](#references)
- [Further Reading](#further-reading)

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Proposed that effective receptive field could be a lot smaller than theoretical receptive field.
* Proposed $L_{2}$ normalization layer to help feature combination/fusion.

----------------------------------------------------------------------------------------------------

## 1 INTRODUCTION

## 2 RELATED WORK

> Exploiting the FCN architecture, ParseNet can directly use global average pooling from the final (or any) feature map, resulting in the feature of the whole image, and use it as context.

## 3 PARSENET

### 3.1 GLOBAL CONTEXT

> Although theoretically, features from the top layers of a network have very large receptive fields (e.g. fc7 in FCN with VGG has a 404x404  pixels receptive field), we argue that in practice, the empirical size of the receptive fields is much smaller, and is not enough to capture the global context.

### 3.2 EARLY FUSION AND LATE FUSION

> Once we get the global context feature, there are two general standard paradigms of using it with the local feature map. First, the early fusion, illustrated in in Fig. 1 where we unpool (replicate) global feature to the same size as of local feature map spatially and then concatenate them, and use the combined feature to learn the classifier. The alternative approach, is late fusion, where each feature is used to learn its own classifier, followed by merging the two predictions into a single classification score Long et al. (2014); Chen et al. (2014).

> Our experiments show that both method works more or less the same if we normalize the feature properly for early fusion case.

### 3.3 $L_2$ NORMALIZATION LAYER

## 4 EXPERIMENTS

## 5 CONCLUSION

----------------------------------------------------------------------------------------------------

## References

* Liu, Wei, Andrew Rabinovich, and Alexander C. Berg. "Parsenet: Looking wider to see better." *arXiv preprint arXiv:1506.04579* (2015).

## Further Reading

* [Long et al. (2014)] FCN
* [Krizhevsky et al. (2012)] AlexNet
* [Szegedy et al. (2014a)] InceptionNetV1
* [Simonyan & Zisserman (2014)] VGGNet
* [Girshick et al. (2014)] R-CNN
* [He et al. (2014)] SPPNet
* [Ioffe & Szegedy (2015)] InceptionNetV2
* [He et al. (2015)] PReLU
