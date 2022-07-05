<span style="font-family:monospace">

# Papers in Computer Vision - Transformer

count: 13

* [Transformer](https://arxiv.org/abs/1706.03762)
    * Title: Attention Is All You Need
    * Year: 12 Jun `2017`
    * Author: Ashish Vaswani
* [Stand-Alone Self-Attention](https://arxiv.org/abs/1906.05909)
    * Title: Stand-Alone Self-Attention in Vision Models
    * Year: 13 Jun `2019`
    * Author: Prajit Ramachandran
* [DETR](https://arxiv.org/abs/2005.12872)
    * Title: End-to-End Object Detection with Transformers
    * Year: 26 May `2020`
    * Author: Nicolas Carion
    * Abstract: We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at [this https URL](https://github.com/facebookresearch/detr).
* [Deformable DETR](https://arxiv.org/abs/2010.04159)
    * Title: Deformable DETR: Deformable Transformers for End-to-End Object Detection
    * Year: 08 Oct `2020`
* [Vision Transformer](https://arxiv.org/abs/2010.11929)
    * Title: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    * Year: 22 Oct `2020`
* [SETR](https://arxiv.org/abs/2012.15840)
    * Title: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers
    * Year: 31 Dec `2020`
* [Swin Transformer](https://arxiv.org/abs/2103.14030)
    * Title: Hierarchical Vision Transformer using Shifted Windows
    * Year: 25 Mar `2021`
* [Swin Transformer V2](https://arxiv.org/abs/2111.09883)
    * Title: Scaling Up Capacity and Resolution
    * Year: 18 Nov `2021`
* [Dynamic Head](https://arxiv.org/abs/2106.08322)
    * Title: Unifying Object Detection Heads with Attentions
    * Year: 15 Jun `2021`
* [Fastformer](https://arxiv.org/abs/2108.09084)
    * Title: Fastformer: Additive Attention Can Be All You Need
    * Year: 20 Aug `2021`
* [MobileViTv1](https://arxiv.org/abs/2110.02178)
    * Title: MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
    * Year: 05 Oct `2021`
    * Author: Sachin Mehta
    * Abstract: Light-weight convolutional neural networks (CNNs) are the de-facto for mobile vision tasks. Their spatial inductive biases allow them to learn representations with fewer parameters across different vision tasks. However, these networks are spatially local. To learn global representations, self-attention-based vision trans-formers (ViTs) have been adopted. Unlike CNNs, ViTs are heavy-weight. In this paper, we ask the following question: is it possible to combine the strengths of CNNs and ViTs to build a light-weight and low latency network for mobile vision tasks? Towards this end, we introduce MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. Our results show that MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters. Our source code is open-source and available at: [this https URL](https://github.com/apple/ml-cvnets).
* [MobileViTv2](https://arxiv.org/abs/2206.02680)
    * Title: Separable Self-attention for Mobile Vision Transformers
    * Year: 06 Jun `2022`
    * Author: Sachin Mehta
    * Abstract: Mobile vision transformers (MobileViT) can achieve state-of-the-art performance across several mobile vision tasks, including classification and detection. Though these models have fewer parameters, they have high latency as compared to convolutional neural network-based models. The main efficiency bottleneck in MobileViT is the multi-headed self-attention (MHA) in transformers, which requires $O(k^{2})$ time complexity with respect to the number of tokens (or patches) $k$. Moreover, MHA requires costly operations (e.g., batch-wise matrix multiplication) for computing self-attention, impacting latency on resource-constrained devices. This paper introduces a separable self-attention method with linear complexity, i.e. $O(k)$. A simple yet effective characteristic of the proposed method is that it uses element-wise operations for computing self-attention, making it a good choice for resource-constrained devices. The improved model, MobileViTv2, is state-of-the-art on several mobile vision tasks, including ImageNet object classification and MS-COCO object detection. With about three million parameters, MobileViTv2 achieves a top-1 accuracy of 75.6% on the ImageNet dataset, outperforming MobileViT by about 1% while running 3.2x faster on a mobile device. Our source code is available at: [this https URL](https://github.com/apple/ml-cvnets).
* [Mask DINO](https://arxiv.org/abs/2206.02777)
    * Title: Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation
    * Year: 06 Jun `2022`
    * Author: Feng Li
    * Abstract: In this paper we present Mask DINO, a unified object detection and segmentation framework. Mask DINO extends DINO (DETR with Improved Denoising Anchor Boxes) by adding a mask prediction branch which supports all image segmentation tasks (instance, panoptic, and semantic). It makes use of the query embeddings from DINO to dot-product a high-resolution pixel embedding map to predict a set of binary masks. Some key components in DINO are extended for segmentation through a shared architecture and training process. Mask DINO is simple, efficient, scalable, and benefits from joint large-scale detection and segmentation datasets. Our experiments show that Mask DINO significantly outperforms all existing specialized segmentation methods, both on a ResNet-50 backbone and a pre-trained model with SwinL backbone. Notably, Mask DINO establishes the best results to date on instance segmentation (54.5 AP on COCO), panoptic segmentation (59.4 PQ on COCO), and semantic segmentation (60.8 mIoU on ADE20K). Code will be avaliable at [this https URL](https://github.com/IDEACVR/MaskDINO).
