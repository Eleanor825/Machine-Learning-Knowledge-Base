# [Notes][Vision][CNN] EfficientNetV2

* url: https://arxiv.org/abs/2104.00298
* Title: EfficientNetV2: Smaller Models and Faster Training
* Year: 01 Apr `2021`
* Authors: Mingxing Tan, Quoc V. Le
* Abstract: This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. To develop this family of models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller. Our training can be further sped up by progressively increasing the image size during training, but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose to adaptively adjust regularization (e.g., dropout and data augmentation) as well, such that we can achieve both fast training and good accuracy. With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources. Code will be available at [this https URL](https://github.com/google/automl/tree/master/efficientnetv2).

----------------------------------------------------------------------------------------------------

## 1. Introduction

Training-aware NAS and scaling

> Our study shows in EfficientNets: (1) training with very large image sizes is slow; (2) depthwise convolutions are slow in early layers. (3) equally scaling up every stage is sub-optimal.

> Based on these observations, we design a search space enriched with additional ops such as Fused-MBConv, and apply training-aware NAS and scaling to jointly optimize model accuracy, training speed, and parameter size.

Progressive learning

> Many previous works, such as progressive resizing (Howard, 2018), FixRes (Touvron et al., 2019), and Mix&Match (Hoffer et al., 2019), have used smaller image sizes in training; however, they usually keep the same regularization for all image sizes, causing a drop in accuracy. We argue that keeping the same regularization for different image sizes is not ideal: for the same network, small image size leads to small network capacity and thus requires weak regularization; vice versa, large image size requires stronger regularization to combat overfitting (see Section 4.1).

> Based on this insight, we propose an improved method of progressive learning: in the early training epochs, we train the network with small image size and weak regularization (e.g., dropout and data augmentation), then we gradually increase image size and add stronger regularization.

Contributions

> Our contributions are threefold:
> * We introduce EfficientNetV2, a new family of smaller and faster models. Found by our training-aware NAS and scaling, EfficientNetV2 outperform previous models in both training speed and parameter efficiency.
> * We propose an improved method of progressive learning, which adaptively adjusts regularization along with image size. We show that it speeds up training, and simultaneously improves accuracy.
> * We demonstrate up to 11x faster training speed and up to 6.8x better parameter efficiency on ImageNet, CIFAR, Cars, and Flowers dataset, than prior art.

## 2. Related work

**Training and Parameter efficiency**

> Many works, such as DenseNet (Huang et al., 2017) and EfficientNet (Tan & Le, 2019a), focus on parameter efficiency, aiming to achieve better accuracy with less parameters.

> RegNet (Radosavovic et al., 2020), ResNeSt (Zhang et al., 2020), TResNet (Ridnik et al., 2020), and EfficientNet-X (Li et al., 2021) focus on GPU and/or TPU inference speed.

> Lambda Networks (Bello, 2021), NFNets (Brock et al., 2021), BoTNets (Srinivas et al., 2021), ResNet-RS (Bello et al., 2021) focus on TPU training speed.

> However, their training speed often comes with the cost of more parameters.

**Progressive Training**

**Neural Architecture Search (NAS)**

> Previous NAS works mostly focus on improving FLOPs efficiency (Tan & Le, 2019b, a) or inference efficiency (Tan et al., 2019; Cai et al., 2019; Wu et al., 2019; Li et al., 2021). Unlike prior works, this paper uses NAS to optimize training and parameter efficiency.

## 3. EfficientNetV2 Architecture Design

### 3.1. Review of EfficientNet

### 3.2. Understanding Training Efficiency

**Training with very large image sizes is slow**

> As pointed out by previous works (Radosavovic et al., 2020), EfficientNet’s large image size results in significant memory usage.

**Depthwise convolutions are slow in early layers**

> Another training bottleneck of EfficientNet comes from the extensive depthwise convolutions (Sifre, 2014).

**Equally scaling up every stage is sub-optimal**

### 3.3. Training-Aware NAS and Scaling

**NAS Search**

> Our training-aware NAS framework is largely based on previous NAS works (Tan et al., 2019; Tan & Le, 2019a), but aims to jointly optimize accuracy, parameter efficiency, and training efficiency on modern accelerators.

**EfficientNetV2 Architecture**

> Compared to the EfficientNet backbone, our searched EfficientNetV2 has several major distinctions:
> 1. The first difference is EfficientNetV2 extensively uses both MBConv (Sandler et al., 2018; Tan & Le, 2019a) and the newly added fused-MBConv (Gupta & Tan, 2019) in the early layers.
> 2. Secondly, EfficientNetV2 prefers smaller expansion ratio for MBConv since smaller expansion ratios tend to have less memory access overhead.
> 3. Thirdly, EfficientNetV2 prefers smaller 3x3 kernel sizes, but it adds more layers to compensate the reduced receptive field resulted from the smaller kernel size.
> 4. Lastly, EfficientNetV2 completely removes the last stride-1 stage in the original EfficientNet, perhaps due to its large parameter size and memory access overhead.

**EfficientNetV2 Scaling**

**Training Speed Comparison**

## 4. Progressive Learning

### 4.1. Motivation

> In addition to FixRes (Touvron et al., 2019), many other works dynamically change image sizes during training (Howard, 2018; Hoffer et al., 2019), but they often cause a drop in accuracy.

> We hypothesize the accuracy drop comes from the unbalanced regularization: when training with different image sizes, we should also adjust the regularization strength accordingly (instead of using a fixed regularization as in previous works).

> In this paper, we argue that even for the same network, smaller image size leads to smaller network capacity and thus needs weaker regularization; vice versa, larger image size leads to more computations with larger capacity, and thus more vulnerable to overfitting.

### 4.2. Progressive Learning with adaptive Regularization

> In the early training epochs, we train the network with smaller images and weak regularization, such that the network can learn simple representations easily and fast. Then, we gradually increase image size but also making learning more difficult by adding stronger regularization.

> Our approach is built upon  (Howard, 2018) that progressively changes image size, but here we adaptively adjust regularization as well.

## 5. Main Results

## 6. Ablation Studies

## 7. Conclusion

----------------------------------------------------------------------------------------------------

## References

* Tan, Mingxing, and Quoc Le. "Efficientnetv2: Smaller models and faster training." *International Conference on Machine Learning*. PMLR, 2021.

## Further Reading

* [Tan & Le, 2019a] EfficientNetV1
* [Huang et al., 2017] DenseNet
