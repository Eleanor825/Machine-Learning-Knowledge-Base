#! https://zhuanlan.zhihu.com/p/566567383
# [Notes][Vision][Transformer] Vision Transformer (ViT)

* url: https://arxiv.org/abs/2010.11929
* Title: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
* Year: 22 Oct `2020`
* Authors: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
* Institutions: [Google Research, Brain Team]
* Abstract: While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

----------------------------------------------------------------------------------------------------

## Summary of Main Contributions

* Applied transformer architecture to image classification.
* Techniques used: class token, patch embedding, position embedding, multi-head self-attention, layer normalization.

----------------------------------------------------------------------------------------------------

## 1 INTRODUCTION

> Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.

> When trained on mid-sized datasets such as ImageNet, such models yield modest accuracies of a few percentage points below ResNets of comparable size. This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.

> However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias.

## 2 RELATED WORK

## 3 METHOD

### 3.1 VISION TRANSFORMER (VIT)

Notations:
* Let $H, W \in \mathbb{Z}_{++}$ denote the height and width of the original image.
* Let $P \in \mathbb{Z}_{++}$ denote the side length of the image patches.
* Let $D \in \mathbb{Z}_{++}$ denote the dimension of the patch embeddings.
* Let $N \in \mathbb{Z}_{++}$ denote the number of patches.
Then $N = HW/P^{2}$.

Then the patch embedding is a process described as follows
$$\mathbb{R}^{H \times W \times C} \overset{\text{reshape}}{\to} \mathbb{R}^{N \times (P^{2}C)} \overset{\text{project}}{\to} \mathbb{R}^{N \times D}$$

> Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.3). The resulting sequence of embedding vectors serves as input to the encoder.

> The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self-attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3).

Notations:
* Let $\textbf{x}^{(1)}, ..., \textbf{x}^{(N)} \in \mathbb{R}^{P^{2}C}$ denote the flattened patches.
* Let $E \in \mathbb{R}^{(P^{2}C) \times D}$ denote the projection used for patch embedding.
* Let $\textbf{x}_{\text{class}} \in \mathbb{R}^{D}$ denote the class token.
* Let $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ denote the positional embedding.

Then
$$\begin{align*}
    \textbf{z}_{0} & := [\textbf{x}_{\text{class}}; \textbf{x}^{(1)}E; ...; \textbf{x}^{(N)}E] + E_{\text{pos}} \\
    \textbf{z}_{l}' & := \operatorname{MSA}(\operatorname{LN}(\textbf{z}_{l-1})) + \textbf{z}_{l-1}, \quad \text{ for } l \in \{1, ..., L\} \\
    \textbf{z}_{l} & := \operatorname{MLP}(\operatorname{LN}(\textbf{z}_{l}')) + \textbf{z}_{l}', \quad \text{ for } l \in \{1, ..., L\} \\
    \textbf{y} & := \operatorname{LN}(\textbf{z}_{L}^{0}).
\end{align*}$$

**Inductive bias**

> We note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution (as described below). Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.

**Hybrid Architecture**

> As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection $E$ (Eq. 1) is applied to patches extracted from a CNN feature map.

> As a special case, the patches can have spatial size 1x1, which means that the input sequence is obtained by simply flattening the spatial dimensions of the feature map and projecting to the Transformer dimension.

### 3.2 FINE-TUNING AND HIGHER RESOLUTION

## 4 EXPERIMENTS

## 5 CONCLUSION

> Unlike prior works using self-attention in computer vision, we do not introduce any image-specific inductive biases into the architecture. Instead, we interpret an image as a sequence of patches and process it by a standard Transformer encoder as used in NLP.

----------------------------------------------------------------------------------------------------

## References

* Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." *arXiv preprint arXiv:2010.11929* (2020).

## Further Reading

* [Krizhevsky et al., 2012] AlexNet
* [He et al., 2016] ResNet
* [Carion et al., 2020] DETR
