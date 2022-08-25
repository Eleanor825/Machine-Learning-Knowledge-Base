# [Notes][Vision][Segmentation] Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials

* url: https://arxiv.org/abs/1210.5644
* Title: Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
* Year: 20 Oct `2012`
* Authors: Philipp Krähenbühl, Vladlen Koltun
* Institutions: Computer Science Department Stanford University
* Abstract: Most state-of-the-art techniques for multi-class image segmentation and labeling use conditional random fields defined over pixels or image regions. While region-level models often feature dense pairwise connectivity, pixel-level models are considerably larger and have only permitted sparse graph structures. In this paper, we consider fully connected CRF models defined on the complete set of pixels in an image. The resulting graphs have billions of edges, making traditional inference algorithms impractical. Our main contribution is a highly efficient approximate inference algorithm for fully connected CRF models in which the pairwise edge potentials are defined by a linear combination of Gaussian kernels. Our experiments demonstrate that dense connectivity at the pixel level substantially improves segmentation and labeling accuracy.

----------------------------------------------------------------------------------------------------

## 1 Introduction

## 2 The Fully Connected CRF Model

Notations:
* Let $k \in \mathbb{Z}_{++}$ denote the number of classes.
* Let $\mathcal{L} := \{l_{1}, ..., l_{k}\}$ denote the labels.
* Let $N \in \mathbb{Z}_{++}$ denote the number of pixels in the input image.
* Let $X_{j} \in \mathcal{L}$ denote the label assigned to pixel $j$ in the input image, for $j \in \{1, ..., N\}$.
* Let $I_{j} \in \mathbb{R}^{3}$ denote the color vector of pixel $j$ in the input image, for $j \in \{1, ..., N\}$.

----------------------------------------------------------------------------------------------------

## References

## Further Reading

