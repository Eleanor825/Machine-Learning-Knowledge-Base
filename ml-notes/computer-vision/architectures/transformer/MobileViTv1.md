# [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

* Year: 05 Oct `2021`
* Author: Sachin Mehta
* Abstract: Light-weight convolutional neural networks (CNNs) are the de-facto for mobile vision tasks. Their spatial inductive biases allow them to learn representations with fewer parameters across different vision tasks. However, these networks are spatially local. To learn global representations, self-attention-based vision trans-formers (ViTs) have been adopted. Unlike CNNs, ViTs are heavy-weight. In this paper, we ask the following question: is it possible to combine the strengths of CNNs and ViTs to build a light-weight and low latency network for mobile vision tasks? Towards this end, we introduce MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. Our results show that MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters. Our source code is open-source and available at: [this https URL](https://github.com/apple/ml-cvnets).

----------------------------------------------------------------------------------------------------

## 1 INTRODUCTION

> Note that floating-point operations (FLOPs) are not sufficient for low latency on mobile devices because FLOPs ignore several important inference-related factors such as memory access, degree of parallelism, and platform characteristics.

> Therefore, instead of optimizing for FLOPs, this paper focuses on designing a light-weight, general-purpose, and low latency network for mobile vision tasks. We achieve this goal with MobileViT that combines the benefits of CNNs (e.g., spatial inductive biases and less sensitivity to data augmentation) and ViTs (e.g., input-adaptive weighting and global processing). Specifically, we introduce the MobileViT block that encodes both local and global information in a tensor effectively. Unlike ViT and its variants (with and without convolutions), MobileViT presents a different perspective to learn global representations. Standard convolution involves three operations: unfolding, local processing, and folding. MobileViT block replaces local processing in convolutions with global processing using transformers. This allows MobileViT block to have CNN- and ViT-like properties, which helps it learn better representations with fewer parameters and simple training recipes (e.g., basic augmentation).

## 2 RELATED WORK

## 3 MOBILEVIT: A LIGHT-WEIGHT TRANSFORMER

> This paper introduces a light-weight ViT model, MobileViT. The core idea is to learn global representations with transformers as convolutions. This allows us to implicitly incorporate convolution-like properties (e.g., spatial bias) in the network, learn representations with simple training recipes (e.g., basic augmentation), and easily integrate MobileViT with downstream architectures (e.e., DeepLabv3 for segmentation).
