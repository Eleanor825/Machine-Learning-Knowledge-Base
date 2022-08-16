# [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)

* Year: 18 Nov `2021`
* Author: Ze Liu
* Abstract: Large-scale NLP models have been shown to significantly improve the performance on language tasks with no signs of saturation. They also demonstrate amazing few-shot capabilities like that of human beings. This paper aims to explore large-scale models in computer vision. We tackle three major issues in training and application of large vision models, including training instability, resolution gaps between pre-training and fine-tuning, and hunger on labelled data. Three main techniques are proposed: 1) a residual-post-norm method combined with cosine attention to improve training stability; 2) A log-spaced continuous position bias method to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs; 3) A self-supervised pre-training method, SimMIM, to reduce the needs of vast labeled images. Through these techniques, this paper successfully trained a 3 billion-parameter Swin Transformer V2 model, which is the largest dense vision model to date, and makes it capable of training with images of up to 1,536$\times$1,536 resolution. It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification. Also note our training is much more efficient than that in Google's billion-level visual models, which consumes 40 times less labelled data and 40 times less training time. Code is available at [this https URL](https://github.com/microsoft/Swin-Transformer).

----------------------------------------------------------------------------------------------------

## 1. Introduction

## 3. Swin Transformer V2
### 3.1. A Brief Review of Swin Transformer

> **Issues in scaling up model capacity and window resolution** We observe two issues when we scale up the capacity and window resolution of the Swin Transformer.
> * *An instability issue when scaling up model capacity*. When we scale up the original Swin Transformer model from small size to large size, the activation values at deeper layers increase dramatically. The  discrepancy  between layers with the highest and the lowest amplitudes has reached an extreme value of $10^{4}$. When we scale it up further to a huge size (658 million parameters), it cannot complete the training.
> * *Degraded performance when transferring models across window resolutions*. The accuracy decreases significantly when we directly test the accuracy of a pre-trained ImageNet-K model ($256 \times 256$ images with $8 \times 8$ window size) at larger image resolutions and window sizes through the bi-cubic interpolation approach. It may be worth re-examining the relative position bias approach in the original Swin Transformer.

### 3.2. Scaling Up Model Capacity

> As mentioned in Section 3.1, the original Swin Transformer (and most vision Transformers) adopts a layer norm layer at the beginning of each block, inherited from vanilla ViT. When we scale up the model capacity, a significant increase in activation values is observed at deeper layers. In fact, in a pre-normalization configuration, the output activation values of each residual block are merged directly back to the main branch, and the amplitude of the main branch grows larger and larger at deeper layers. Large amplitude discrepancy in different layers causes training instability.

> **Post normalization** To ease this problem, we propose too use a *residual post normalization* approach instead. In this approach, the output of  each residual block is normalized before merging back into the main branch, and the amplitude of the main branch does not accumulate when the layer goes deeper. The activation amplitudes by this approach are much milder than in the original pre-normalization configuration.
> In our largest model training, we introduce an additional layer normalization layer on the main branch every 6 Transformer blocks, to further stabilize training.

> **Scaled cosine attention** In the original self-attention computation, the similarity terms of the pixel pairs are computed as a dot product of the query and key vectors. We find that when this approach is used in large visual models, the learnt attention maps of some blocks and heads are frequently dominated by a few pixel pairs, especially in the res-post-norm configuration. To ease this issue, we propose a scaled cosine attention approach that computes the attention logit of a pixel pair $i$ adn $j$ by a scaled cosine function:
> $$\text{Sim}(\textbf{q}_{i}, \textbf{k}_{j}) = \cos(\textbf{q}_{i}, \textbf{k}_{j})/\tau + B_{ij}$$
> where $B_{ij}$ is the relative position bias between pixel $i$ and $j$; $\tau$ is a learnabe scalar, non-shared across heads and layers.
