# [Early Convolutions Help Transformers See Better](https://arxiv.org/abs/2106.14881)

* Year: 28 Jun `2021`
* Author: Tete Xiao
* Abstract: Vision transformer (ViT) models exhibit substandard optimizability. In particular, they are sensitive to the choice of optimizer (AdamW vs. SGD), optimizer hyperparameters, and training schedule length. In comparison, modern convolutional neural networks are easier to optimize. Why is this the case? In this work, we conjecture that the issue lies with the patchify stem of ViT models, which is implemented by a stride-p pxp convolution (p=16 by default) applied to the input image. This large-kernel plus large-stride convolution runs counter to typical design choices of convolutional layers in neural networks. To test whether this atypical design choice causes an issue, we analyze the optimization behavior of ViT models with their original patchify stem versus a simple counterpart where we replace the ViT stem by a small number of stacked stride-two 3x3 convolutions. While the vast majority of computation in the two ViT designs is identical, we find that this small change in early visual processing results in markedly different training behavior in terms of the sensitivity to optimization settings as well as the final model accuracy. Using a convolutional stem in ViT dramatically increases optimization stability and also improves peak performance (by ~1-2% top-1 accuracy on ImageNet-1k), while maintaining flops and runtime. The improvement can be observed across the wide spectrum of model complexities (from 1G to 36G flops) and dataset scales (from ImageNet-1k to ImageNet-21k). These findings lead us to recommend using a standard, lightweight convolutional stem for ViT models in this regime as a more robust architectural choice compared to the original ViT model design.

----------------------------------------------------------------------------------------------------

## 1 Introduction

> Vision transformer (ViT) models offer an alternative design paradigm to convolutional neural networks (CNNs). ViTs replace the inductive bias towards local processing inherent in convolutions with global processing performed by multi-headed self-attention. The hope is that this design has the potential to improve performance on vision tasks, akin to the trends observed in natural language processing.

> ViT models exhibit substandard optimizability. ViTs are sensitive to the choice of optimizer, to the selection of dataset specific learning hyperparameters, to training schedule length, to network depth, etc. These issues render former training recipes and intuitions ineffective and impede research.

> Convolutional neural networks, in contrast, are exceptionally easy and robust to optimize. Simple training recipes based onSGD, basic data augmentation, and standard hyperparameter values have been widely used for years.

> In this paper we hypothesize that the issues lies primarily in the early visual processing performed by ViT. ViT "patchifies" the input image into $p \times p$ non-overlapping patches to form the transformer encoder's input set. This patchify stem is implemented as a stride-$p$ $p \times p$ convolution, with $p = 16$ as a default value

> To test this hypothesis, we minimally change the early visual processing of ViT by replacing its patchify stem with a standard convolutional stem consisting of only ~5 convolutions. To compensate for the small addition in flops, we remove one transformer block to maintain parity in flops and runtime.

> In extensive experiments we show that replacing the ViT patchify stem with a more standard convolutional stem (i) allows ViT to converge faster, (ii) enables, for the first time, the use of either AdamW or SGD without a significant drop in accuracy, (iii) brings ViT's stability w.r.t. learning rate and weight decay closer to that of modern CNNs, and (iv) yields improvements in ImageNet top-1 error of ~-2 percentage points.

> The results show that injecting some convolutional inductive bias into ViTs can be beneficial under commonly studied settings.

> We conjecture that restricting convolutions in ViT to early visual processing may be a crucial design choice that strikes a balance between (hard) inductive biases and the representation learning ability of transformer blocks.
