# [Notes][Vision][Segmentation] Panoptic FPN

* url: https://arxiv.org/abs/1901.02446
* Title: Panoptic Feature Pyramid Networks
* Year: 08 Jan `2019`
* Authors: Alexander Kirillov, Ross Girshick, Kaiming He, Piotr Dollár
* Institutions: [Facebook AI Research (FAIR)]
* Abstract: The recently introduced panoptic segmentation task has renewed our community's interest in unifying the tasks of instance segmentation (for thing classes) and semantic segmentation (for stuff classes). However, current state-of-the-art methods for this joint task use separate and dissimilar networks for instance and semantic segmentation, without performing any shared computation. In this work, we aim to unify these methods at the architectural level, designing a single network for both tasks. Our approach is to endow Mask R-CNN, a popular instance segmentation method, with a semantic segmentation branch using a shared Feature Pyramid Network (FPN) backbone. Surprisingly, this simple baseline not only remains effective for instance segmentation, but also yields a lightweight, top-performing method for semantic segmentation. In this work, we perform a detailed study of this minimally extended version of Mask R-CNN with FPN, which we refer to as Panoptic FPN, and show it is a robust and accurate baseline for both tasks. Given its effectiveness and conceptual simplicity, we hope our method can serve as a strong baseline and aid future research in panoptic segmentation.

----------------------------------------------------------------------------------------------------

## 1. Introduction

## 2. Related Work

----------------------------------------------------------------------------------------------------

## References

* Kirillov, Alexander, et al. "Panoptic feature pyramid networks." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.

## Further Reading

* [2] [SegNet](https://zhuanlan.zhihu.com/p/568804052)
* [7] Cascade R-CNN
* [12] DeepLabv3+
* [15] Deformable ConvNets v1
* [19] Laplacian Pyramid
* [20] Fast R-CNN
* [21] R-CNN
* [23] Mask R-CNN
* [25] Recombinator Networks
* [29] Panoptic Segmentation
* [34] Feature Pyramid Networks
* [46] Faster R-CNN
* [47] [U-Net](https://zhuanlan.zhihu.com/p/568803926)
* [53] ResNeXt
* [55] Dilated Convolutions
