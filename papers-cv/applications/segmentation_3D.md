# [Papers][Vision] 3D Segmentation <!-- omit in toc -->

count=1

## Table of Contents <!-- omit in toc -->

- [Unknown](#unknown)

----------------------------------------------------------------------------------------------------

## Unknown

* [[V-Net](https://arxiv.org/abs/1606.04797)]
    [[pdf](https://arxiv.org/pdf/1606.04797.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1606.04797/)]
    * Title: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    * Year: 15 Jun `2016`
    * Authors: Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi
    * Abstract: Convolutional Neural Networks (CNNs) have been recently employed to solve problems from both the computer vision and medical image analysis fields. Despite their popularity, most approaches are only able to process 2D images while most medical data used in clinical practice consists of 3D volumes. In this work we propose an approach to 3D image segmentation based on a volumetric, fully convolutional, neural network. Our CNN is trained end-to-end on MRI volumes depicting prostate, and learns to predict segmentation for the whole volume at once. We introduce a novel objective function, that we optimise during training, based on Dice coefficient. In this way we can deal with situations where there is a strong imbalance between the number of foreground and background voxels. To cope with the limited number of annotated volumes available for training, we augment the data applying random non-linear transformations and histogram matching. We show in our experimental evaluation that our approach achieves good performances on challenging test data while requiring only a fraction of the processing time needed by other previous methods.
