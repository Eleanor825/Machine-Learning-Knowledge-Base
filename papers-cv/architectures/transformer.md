# [Papers][Vision] Transformer Architectures <!-- omit in toc -->

count=80

## Table of Contents <!-- omit in toc -->

- [Basics](#basics)
- [Unknown](#unknown)
- [Self-attention in vision models (4)](#self-attention-in-vision-models-4)
- [Incorporate Attention Module into CNN (2021, PVT) (count=3)](#incorporate-attention-module-into-cnn-2021-pvt-count3)
- [Improving self-attention (4 + 3 + 2)](#improving-self-attention-4--3--2)
- [Scaling up of vision models (3 + 5 + 3)](#scaling-up-of-vision-models-3--5--3)
- [ResNet-Like Architectures (2020, ViT) (count=3)](#resnet-like-architectures-2020-vit-count3)
- [Vision Transformer Variants at Relatively Small Scale (11)](#vision-transformer-variants-at-relatively-small-scale-11)
- [Multi-scale networks (5)](#multi-scale-networks-5)
- [Locality Prior (5)](#locality-prior-5)
- [Referenced by Swin Transformer V1](#referenced-by-swin-transformer-v1)
  - [(2021, Swin Transformer V1) Self-attention based backbone architectures (count=3)](#2021-swin-transformer-v1-self-attention-based-backbone-architectures-count3)
  - [(2021, Swin Transformer V1) Self-attention/Transformers to complement CNN Backbones (count=6)](#2021-swin-transformer-v1-self-attentiontransformers-to-complement-cnn-backbones-count6)
  - [(2021, Swin Transformer V1) Self-attention/Transformers to complement CNN heads (count=2)](#2021-swin-transformer-v1-self-attentiontransformers-to-complement-cnn-heads-count2)
  - [(2021, Swin Transformer V1) Transformer based vision backbones (count=5)](#2021-swin-transformer-v1-transformer-based-vision-backbones-count5)
- [Combine convolutions and transformers (3 + 1)](#combine-convolutions-and-transformers-3--1)
- [Improving Transformer-Based Models (MobileViTv2)](#improving-transformer-based-models-mobilevitv2)
- [Continuous convolution and variants (4)](#continuous-convolution-and-variants-4)
- [Sliding Window Approaches](#sliding-window-approaches)
- [Dictionary lookup problem (2021, PVT) (count=5)](#dictionary-lookup-problem-2021-pvt-count5)
- [Substandard Optimizability is Due to the Lack of Spatial Inductive Biases in ViTs.](#substandard-optimizability-is-due-to-the-lack-of-spatial-inductive-biases-in-vits)
- [Transformer Architecture Applied to Object Detection and Instance Segmentation (4)](#transformer-architecture-applied-to-object-detection-and-instance-segmentation-4)
- [MobileViT](#mobilevit)
- [Swin Transformer](#swin-transformer)
- [Others](#others)

----------------------------------------------------------------------------------------------------

> (2021, Transformers in Vision) There exist two key ideas that have contributed towards the development of transformer models.
> (a) The first one is self-supervision, which is used to pre-train transformer models on a large unlabeled corpus, subsequently fine-tuning them to the target task with a small labeled dataset [35, 96, 150].
> (b) The second key idea is that of self-attention which allows capturing 'long-term' information and dependencies between sequence elements as compared to conventional recurrent models that find it challenging to encode such relationships.

## Basics

* [[Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1705.03122.pdf)]
    [vanity]
    * Title: Convolutional Sequence to Sequence Learning
    * Year: 08 May `2017`
    * Authors: Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin
    * Abstract: The prevalent approach to sequence to sequence learning maps an input sequence to a variable length output sequence via recurrent neural networks. We introduce an architecture based entirely on convolutional neural networks. Compared to recurrent models, computations over all elements can be fully parallelized during training and optimization is easier since the number of non-linearities is fixed and independent of the input length. Our use of gated linear units eases gradient propagation and we equip each decoder layer with a separate attention module. We outperform the accuracy of the deep LSTM setup of Wu et al. (2016) on both WMT'14 English-German and WMT'14 English-French translation at an order of magnitude faster speed, both on GPU and CPU.
    * Comments:
        * (2017, Transformer) introduced learned positional embeddings.
* [[Transformer](https://arxiv.org/abs/1706.03762)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1706.03762.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1706.03762/)]
    * Title: Attention Is All You Need
    * Year: 12 Jun `2017`
    * Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    * Institutions: [Google Brain], [Google Research], [University of Toronto]
    * Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
    * Comments:
        * > (2022, Recent Advances) This transformer is solely based on attention mechanism, instead of convolution layers.
* [[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2010.11929.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2010.11929/)]
    * Title: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    * Year: 22 Oct `2020`
    * Authors: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
    * Institutions: [Google Research, Brain Team]
    * Abstract: While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
    * Comments:
        * > Vision Transformers (Dosovitskiy et al., 2021) improves training efficiency on large-scale datasets by using Transformer blocks. (EfficientNetV2, 2021)
        * > ViT first partitions an input image into non-overlapping $p \times p$ patches and linearly projects each patch to a $d$-dimensional feature vector using a learned weight matrix. A patch size of $p = 16$ and an image size of $224 \times 224$ are typical. The resulting patch embeddings (plus positional embeddings and a learned classification token embedding) are processed by a standard transformer encoder followed by a classification head. Using common network nomenclature, we refer to the portion of ViT before the transformer blocks as the network's stem. ViT's stem is a specific case of convolution (stride-$p$, $p \times p$ kernel), but we will refer to it as the patchify stem and reserve the terminology of convolutional stem for stems with a more conventional CNN design with multiple layers of overlapping convolutions (i.e., with stride smaller tha the kernel size). (Early Convolutions Help Transformers See Better, 2021)
        * > (2021, PVT) ViT has a columnar structure with coarse image patches (i.e., dividing image with a large patch size) as input. Although ViT is applicable to image classification, it is challenging to be directly adapted to pixel-level dense predictions, e.g., object detection and segmentation, because (1) its output feature map has only a single scale with low resolution and (2) its computations and memory cost are relatively high even for common input image size (e.g., shorter edge of 800 pixels in COCO detection benchmark).
        * > (2021, PVT) Similar to the traditional Transformer [51], the length of ViT’s output sequence is the same as the input, which means that the output of ViT is single-scale (see Figure 1 (b)).
        * > (2021, PVT) Due to the limited resource., the output of ViT is coarse-grained (e.g., the patch size is 16 or 32 pixels), and thus its output resolution is relatively low (e.g., 16-stride or 32-stride). As a result, it is difficult to directly apply ViT in dense prediction tasks that require high-resolution or multi-scale feature maps.
        * > (2021, Swin Transformer V1) The pioneering work of ViT directly applies a Transformer architecture on non-overlapping medium-sized image patches for image classification. It achieves an impressive speed-accuracy trade-off on image classification compared to convolutional networks.

## Unknown

* [[DSTT](https://arxiv.org/abs/2104.06637)]
    [[pdf](https://arxiv.org/pdf/2104.06637.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2104.06637/)]
    * Title: Decoupled Spatial-Temporal Transformer for Video Inpainting
    * Year: 14 Apr `2021`
    * Authors: Rui Liu, Hanming Deng, Yangyi Huang, Xiaoyu Shi, Lewei Lu, Wenxiu Sun, Xiaogang Wang, Jifeng Dai, Hongsheng Li
    * Abstract: Video inpainting aims to fill the given spatiotemporal holes with realistic appearance but is still a challenging task even with prosperous deep learning approaches. Recent works introduce the promising Transformer architecture into deep video inpainting and achieve better performance. However, it still suffers from synthesizing blurry texture as well as huge computational cost. Towards this end, we propose a novel Decoupled Spatial-Temporal Transformer (DSTT) for improving video inpainting with exceptional efficiency. Our proposed DSTT disentangles the task of learning spatial-temporal attention into 2 sub-tasks: one is for attending temporal object movements on different frames at same spatial locations, which is achieved by temporally-decoupled Transformer block, and the other is for attending similar background textures on same frame of all spatial positions, which is achieved by spatially-decoupled Transformer block. The interweaving stack of such two blocks makes our proposed model attend background textures and moving objects more precisely, and thus the attended plausible and temporally-coherent appearance can be propagated to fill the holes. In addition, a hierarchical encoder is adopted before the stack of Transformer blocks, for learning robust and hierarchical features that maintain multi-level local spatial structure, resulting in the more representative token vectors. Seamless combination of these two novel designs forms a better spatial-temporal attention scheme and our proposed model achieves better performance than state-of-the-art video inpainting approaches with significant boosted efficiency.
* [[Rethinking and Improving Relative Position Encoding for Vision Transformer](https://arxiv.org/abs/2107.14222)]
    [[pdf](https://arxiv.org/pdf/2107.14222.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2107.14222/)]
    * Title: Rethinking and Improving Relative Position Encoding for Vision Transformer
    * Year: 29 Jul `2021`
    * Authors: Kan Wu, Houwen Peng, Minghao Chen, Jianlong Fu, Hongyang Chao
    * Abstract: Relative position encoding (RPE) is important for transformer to capture sequence ordering of input tokens. General efficacy has been proven in natural language processing. However, in computer vision, its efficacy is not well studied and even remains controversial, e.g., whether relative position encoding can work equally well as absolute position? In order to clarify this, we first review existing relative position encoding methods and analyze their pros and cons when applied in vision transformers. We then propose new relative position encoding methods dedicated to 2D images, called image RPE (iRPE). Our methods consider directional relative distance modeling as well as the interactions between queries and relative position embeddings in self-attention mechanism. The proposed iRPE methods are simple and lightweight. They can be easily plugged into transformer blocks. Experiments demonstrate that solely due to the proposed encoding methods, DeiT and DETR obtain up to 1.5% (top-1 Acc) and 1.3% (mAP) stable improvements over their original versions on ImageNet and COCO respectively, without tuning any extra hyperparameters such as learning rate and weight decay. Our ablation and analysis also yield interesting findings, some of which run counter to previous understanding. Code and models are open-sourced at this https URL.
* [[SimMIM](https://arxiv.org/abs/2111.09886)]
    [[pdf](https://arxiv.org/pdf/2111.09886.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2111.09886/)]
    * Title: SimMIM: A Simple Framework for Masked Image Modeling
    * Year: 18 Nov `2021`
    * Authors: Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, Han Hu
    * Abstract: This paper presents SimMIM, a simple framework for masked image modeling. We simplify recently proposed related approaches without special designs such as block-wise masking and tokenization via discrete VAE or clustering. To study what let the masked image modeling task learn good representations, we systematically study the major components in our framework, and find that simple designs of each component have revealed very strong representation learning performance: 1) random masking of the input image with a moderately large masked patch size (e.g., 32) makes a strong pre-text task; 2) predicting raw pixels of RGB values by direct regression performs no worse than the patch classification approaches with complex designs; 3) the prediction head can be as light as a linear layer, with no worse performance than heavier ones. Using ViT-B, our approach achieves 83.8% top-1 fine-tuning accuracy on ImageNet-1K by pre-training also on this dataset, surpassing previous best approach by +0.6%. When applied on a larger model of about 650 million parameters, SwinV2-H, it achieves 87.1% top-1 accuracy on ImageNet-1K using only ImageNet-1K data. We also leverage this approach to facilitate the training of a 3B model (SwinV2-G), that by $40\times$ less data than that in previous practice, we achieve the state-of-the-art on four representative vision benchmarks. The code and models will be publicly available at this https URL.
* [[Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)]
    [[pdf](https://arxiv.org/pdf/2103.17239.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.17239/)]
    * Title: Going deeper with Image Transformers
    * Year: 31 Mar `2021`
    * Authors: Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, Hervé Jégou
    * Abstract: Transformers have been recently adapted for large scale image classification, achieving high scores shaking up the long supremacy of convolutional neural networks. However the optimization of image transformers has been little studied so far. In this work, we build and optimize deeper transformer networks for image classification. In particular, we investigate the interplay of architecture and optimization of such dedicated transformers. We make two transformers architecture changes that significantly improve the accuracy of deep transformers. This leads us to produce models whose performance does not saturate early with more depth, for instance we obtain 86.5% top-1 accuracy on Imagenet when training with no external data, we thus attain the current SOTA with less FLOPs and parameters. Moreover, our best model establishes the new state of the art on Imagenet with Reassessed labels and Imagenet-V2 / match frequency, in the setting with no additional training data. We share our code and models.
* [[DeepViT](https://arxiv.org/abs/2103.11886)]
    [[pdf](https://arxiv.org/pdf/2103.11886.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.11886/)]
    * Title: DeepViT: Towards Deeper Vision Transformer
    * Year: 22 Mar `2021`
    * Authors: Daquan Zhou, Bingyi Kang, Xiaojie Jin, Linjie Yang, Xiaochen Lian, Zihang Jiang, Qibin Hou, Jiashi Feng
    * Abstract: Vision transformers (ViTs) have been successfully applied in image classification tasks recently. In this paper, we show that, unlike convolution neural networks (CNNs)that can be improved by stacking more convolutional layers, the performance of ViTs saturate fast when scaled to be deeper. More specifically, we empirically observe that such scaling difficulty is caused by the attention collapse issue: as the transformer goes deeper, the attention maps gradually become similar and even much the same after certain layers. In other words, the feature maps tend to be identical in the top layers of deep ViT models. This fact demonstrates that in deeper layers of ViTs, the self-attention mechanism fails to learn effective concepts for representation learning and hinders the model from getting expected performance gain. Based on above observation, we propose a simple yet effective method, named Re-attention, to re-generate the attention maps to increase their diversity at different layers with negligible computation and memory cost. The pro-posed method makes it feasible to train deeper ViT models with consistent performance improvements via minor modification to existing ViT models. Notably, when training a deep ViT model with 32 transformer blocks, the Top-1 classification accuracy can be improved by 1.6% on ImageNet. Code is publicly available at this https URL.
    * Comments:
        * (2022, Recent Advances) In CNN architectures, we can easily increase performance by stacking more convolutional layers but transformers are different which are quickly saturated when architecture becomes deeper. The reason is that as the transformer enters the deep layer, the attention map becomes more and more similar. Based on this, Zhou et al. (2021) introduced a re-attention module, which regenerated the attention map with a little computational cost in order to enhance the diversity between layers.
* [[Delight](https://arxiv.org/abs/2008.00623)]
    [[pdf](https://arxiv.org/pdf/2008.00623.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2008.00623/)]
    * Title: DeLighT: Deep and Light-weight Transformer
    * Year: 03 Aug `2020`
    * Authors: Sachin Mehta, Marjan Ghazvininejad, Srinivasan Iyer, Luke Zettlemoyer, Hannaneh Hajishirzi
    * Abstract: We introduce a deep and light-weight transformer, DeLighT, that delivers similar or better performance than standard transformer-based models with significantly fewer parameters. DeLighT more efficiently allocates parameters both (1) within each Transformer block using the DeLighT transformation, a deep and light-weight transformation, and (2) across blocks using block-wise scaling, which allows for shallower and narrower DeLighT blocks near the input and wider and deeper DeLighT blocks near the output. Overall, DeLighT networks are 2.5 to 4 times deeper than standard transformer models and yet have fewer parameters and operations. Experiments on benchmark machine translation and language modeling tasks show that DeLighT matches or improves the performance of baseline Transformers with 2 to 3 times fewer parameters on average. Our source code is available at: \url{this https URL}
* [[LambdaNetworks](https://arxiv.org/abs/2102.08602)]
    [[pdf](https://arxiv.org/pdf/2102.08602.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2102.08602/)]
    * Title: LambdaNetworks: Modeling Long-Range Interactions Without Attention
    * Year: 17 Feb `2021`
    * Authors: Irwan Bello
    * Abstract: We present lambda layers -- an alternative framework to self-attention -- for capturing long-range interactions between an input and structured contextual information (e.g. a pixel surrounded by other pixels). Lambda layers capture such interactions by transforming available contexts into linear functions, termed lambdas, and applying these linear functions to each input separately. Similar to linear attention, lambda layers bypass expensive attention maps, but in contrast, they model both content and position-based interactions which enables their application to large structured inputs such as images. The resulting neural network architectures, LambdaNetworks, significantly outperform their convolutional and attentional counterparts on ImageNet classification, COCO object detection and COCO instance segmentation, while being more computationally efficient. Additionally, we design LambdaResNets, a family of hybrid architectures across different scales, that considerably improves the speed-accuracy tradeoff of image classification models. LambdaResNets reach excellent accuracies on ImageNet while being 3.2 - 4.4x faster than the popular EfficientNets on modern machine learning accelerators. When training with an additional 130M pseudo-labeled images, LambdaResNets achieve up to a 9.5x speed-up over the corresponding EfficientNet checkpoints.
    * Comments:
        * > Lambda Networks (Bello, 2021) and BotNet (Srinivas et al., 2021) improve training speed by using attention layers in ConvNets. (EfficientNetV2, 2021)
        * > Lambda Networks (Bello, 2021), NFNets (Brock et al., 2021), BoTNets (Srinivas et al., 2021), ResNet-RS (Bello et al., 2021) focus on TPU training speed. (EfficientNetV2, 2021)
* [[CrossViT](https://arxiv.org/abs/2103.14899)]
    [[pdf](https://arxiv.org/pdf/2103.14899.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.14899/)]
    * Title: CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification
    * Year: 27 Mar `2021`
    * Authors: Chun-Fu Chen, Quanfu Fan, Rameswar Panda
    * Abstract: The recently developed vision transformer (ViT) has achieved promising results on image classification compared to convolutional neural networks. Inspired by this, in this paper, we study how to learn multi-scale feature representations in transformer models for image classification. To this end, we propose a dual-branch transformer to combine image patches (i.e., tokens in a transformer) of different sizes to produce stronger image features. Our approach processes small-patch and large-patch tokens with two separate branches of different computational complexity and these tokens are then fused purely by attention multiple times to complement each other. Furthermore, to reduce computation, we develop a simple yet effective token fusion module based on cross attention, which uses a single token for each branch as a query to exchange information with other branches. Our proposed cross-attention only requires linear time for both computational and memory complexity instead of quadratic time otherwise. Extensive experiments demonstrate that our approach performs better than or on par with several concurrent works on vision transformer, in addition to efficient CNN models. For example, on the ImageNet1K dataset, with some architectural changes, our approach outperforms the recent DeiT by a large margin of 2\% with a small to moderate increase in FLOPs and model parameters. Our source codes and models are available at \url{this https URL}.
    * Comments:
        * > (2022, Recent Advances) Chen et al. (2021a) introduced dual-branch ViT to extract multi-scale feature representations and developed a cross-attention-based token-fusion mechanism, which is linear in terms of memory and computation to combine features at different scales.
* [[AANet](https://arxiv.org/abs/1904.09925)]
    [[pdf](https://arxiv.org/pdf/1904.09925.pdf)]
    [vanity]
    * Title: Attention Augmented Convolutional Networks
    * Year: 22 Apr `2019`
    * Authors: Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, Quoc V. Le
    * Abstract: Convolutional networks have been the paradigm of choice in many computer vision applications. The convolution operation however has a significant weakness in that it only operates on a local neighborhood, thus missing global information. Self-attention, on the other hand, has emerged as a recent advance to capture long range interactions, but has mostly been applied to sequence modeling and generative modeling tasks. In this paper, we consider the use of self-attention for discriminative visual tasks as an alternative to convolutions. We introduce a novel two-dimensional relative self-attention mechanism that proves competitive in replacing convolutions as a stand-alone computational primitive for image classification. We find in control experiments that the best results are obtained when combining both convolutions and self-attention. We therefore propose to augment convolutional operators with this self-attention mechanism by concatenating convolutional feature maps with a set of feature maps produced via self-attention. Extensive experiments show that Attention Augmentation leads to consistent improvements in image classification on ImageNet and object detection on COCO across many different models and scales, including ResNets and a state-of-the art mobile constrained network, while keeping the number of parameters similar. In particular, our method achieves a $1.3\%$ top-1 accuracy improvement on ImageNet classification over a ResNet50 baseline and outperforms other attention mechanisms for images such as Squeeze-and-Excitation. It also achieves an improvement of 1.4 mAP in COCO Object Detection on top of a RetinaNet baseline.
* [[EAANet](https://arxiv.org/abs/2206.01821)]
    [[pdf](https://arxiv.org/pdf/2206.01821.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2206.01821/)]
    * Title: EAANet: Efficient Attention Augmented Convolutional Networks
    * Year: 03 Jun `2022`
    * Authors: Runqing Zhang, Tianshu Zhu
    * Abstract: Humans can effectively find salient regions in complex scenes. Self-attention mechanisms were introduced into Computer Vision (CV) to achieve this. Attention Augmented Convolutional Network (AANet) is a mixture of convolution and self-attention, which increases the accuracy of a typical ResNet. However, The complexity of self-attention is O(n2) in terms of computation and memory usage with respect to the number of input tokens. In this project, we propose EAANet: Efficient Attention Augmented Convolutional Networks, which incorporates efficient self-attention mechanisms in a convolution and self-attention hybrid architecture to reduce the model's memory footprint. Our best model show performance improvement over AA-Net and ResNet18. We also explore different methods to augment Convolutional Network with self-attention mechanisms and show the difficulty of training those methods compared to ResNet. Finally, we show that augmenting efficient self-attention mechanisms with ResNet scales better with input size than normal self-attention mechanisms. Therefore, our EAANet is more capable of working with high-resolution images.
* [[Music Transformer](https://arxiv.org/abs/1809.04281)]
    [[pdf](https://arxiv.org/pdf/1809.04281.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1809.04281/)]
    * Title: Music Transformer
    * Year: 12 Sep `2018`
    * Authors: Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, Ian Simon, Curtis Hawthorne, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, Douglas Eck
    * Abstract: Music relies heavily on repetition to build structure and meaning. Self-reference occurs on multiple timescales, from motifs to phrases to reusing of entire sections of music, such as in pieces with ABA structure. The Transformer (Vaswani et al., 2017), a sequence model based on self-attention, has achieved compelling results in many generation tasks that require maintaining long-range coherence. This suggests that self-attention might also be well-suited to modeling music. In musical composition and performance, however, relative timing is critically important. Existing approaches for representing relative positional information in the Transformer modulate attention based on pairwise distance (Shaw et al., 2018). This is impractical for long sequences such as musical compositions since their memory complexity for intermediate relative information is quadratic in the sequence length. We propose an algorithm that reduces their intermediate memory requirement to linear in the sequence length. This enables us to demonstrate that a Transformer with our modified relative attention mechanism can generate minute-long compositions (thousands of steps, four times the length modeled in Oore et al., 2018) with compelling structure, generate continuations that coherently elaborate on a given motif, and in a seq2seq setup generate accompaniments conditioned on melodies. We evaluate the Transformer with our relative attention mechanism on two datasets, JSB Chorales and Piano-e-Competition, and obtain state-of-the-art results on the latter.
* [[The Evolved Transformer](https://arxiv.org/abs/1901.11117)]
    [[pdf](https://arxiv.org/pdf/1901.11117.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1901.11117/)]
    * Title: The Evolved Transformer
    * Year: 30 Jan `2019`
    * Authors: David R. So, Chen Liang, Quoc V. Le
    * Abstract: Recent works have highlighted the strength of the Transformer architecture on sequence tasks while, at the same time, neural architecture search (NAS) has begun to outperform human-designed models. Our goal is to apply NAS to search for a better alternative to the Transformer. We first construct a large search space inspired by the recent advances in feed-forward sequence models and then run evolutionary architecture search with warm starting by seeding our initial population with the Transformer. To directly search on the computationally expensive WMT 2014 English-German translation task, we develop the Progressive Dynamic Hurdles method, which allows us to dynamically allocate more resources to more promising candidate models. The architecture found in our experiments -- the Evolved Transformer -- demonstrates consistent improvement over the Transformer on four well-established language tasks: WMT 2014 English-German, WMT 2014 English-French, WMT 2014 English-Czech and LM1B. At a big model size, the Evolved Transformer establishes a new state-of-the-art BLEU score of 29.8 on WMT'14 English-German; at smaller sizes, it achieves the same quality as the original "big" Transformer with 37.6% less parameters and outperforms the Transformer by 0.7 BLEU at a mobile-friendly model size of 7M parameters.

## Self-attention in vision models (4)

* [[On the Relationship between Self-Attention and Convolutional Layers](https://arxiv.org/abs/1911.03584)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1911.03584.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1911.03584/)]
    * Title: On the Relationship between Self-Attention and Convolutional Layers
    * Year: 08 Nov `2019`
    * Authors: Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi
    * Abstract: Recent trends of incorporating attention mechanisms in vision have led researchers to reconsider the supremacy of convolutional layers as a primary building block. Beyond helping CNNs to handle long-range dependencies, Ramachandran et al. (2019) showed that attention can completely replace convolution and achieve state-of-the-art performance on vision tasks. This raises the question: do learned attention layers operate similarly to convolutional layers? This work provides evidence that attention layers can perform convolution and, indeed, they often learn to do so in practice. Specifically, we prove that a multi-head self-attention layer with sufficient number of heads is at least as expressive as any convolutional layer. Our numerical experiments then show that self-attention layers attend to pixel-grid patterns similarly to CNN layers, corroborating our analysis. Our code is publicly available.
    * Comments:
        * > Transformers are revolutionizing natural language processing by enabling scalable training. Transformers use multi-headed self-attention, which performs global information processing and is strictly more general than convolution. (Early Convolutions Help Transformers See Better, 2021)
    
## Incorporate Attention Module into CNN (2021, PVT) (count=3)

* [[Non-local Neural Networks](https://ieeexplore.ieee.org/document/8578911)] <!-- printed -->
    * Title: Non-local Neural Networks
    * Year: 16 December `2018`
    * Author: Xiaolong Wang
    * Abstract: Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time. In this paper, we present non-local operations as a generic family of building blocks for capturing long-range dependencies. Inspired by the classical non-local means method [4] in computer vision, our non-local operation computes the response at a position as a weighted sum of the features at all positions. This building block can be plugged into many computer vision architectures. On the task of video classification, even without any bells and whistles, our nonlocal models can compete or outperform current competition winners on both Kinetics and Charades datasets. In static image recognition, our non-local models improve object detection/segmentation and pose estimation on the COCO suite of tasks. Code will be made available.
    * Comments:
        * > (2020, ViT) Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019, Wang et al., 2020). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns.
        * > (2021, Early Convolutions) Wang et al. show that (single-headed) self-attention is a form of non-local means and that integrating it into a ResNet improves several tasks. Ramachandran et al. explore this direction further with stand-alone self-attention networks for vision. They report difficulties in designing an attention-based network stem and present a bespoke solution that avoids convolutions.
        * > (2021, PVT) The non-local block attempts to model long-range dependencies in both space and time, which has been shown beneficial for accurate video classification.
        * > (2019, AANet) In non-local neural networks [45], improvements are shown in video classification and object detection via the additive use of a few non-local residual blocks that employ self-attention in convolutional architectures.
* [[Stand-Alone Self-Attention](https://arxiv.org/abs/1906.05909)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/1906.05909.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1906.05909/)]
    * Title: Stand-Alone Self-Attention in Vision Models
    * Year: 13 Jun `2019`
    * Authors: Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, Jonathon Shlens
    * Abstract: Convolutions are a fundamental building block of modern computer vision systems. Recent approaches have argued for going beyond convolutions in order to capture long-range dependencies. These efforts focus on augmenting convolutional models with content-based interactions, such as self-attention and non-local means, to achieve gains on a number of vision tasks. The natural question that arises is whether attention can be a stand-alone primitive for vision models instead of serving as just an augmentation on top of convolutions. In developing and testing a pure self-attention vision model, we verify that self-attention can indeed be an effective stand-alone layer. A simple procedure of replacing all instances of spatial convolutions with a form of self-attention applied to ResNet model produces a fully self-attentional model that outperforms the baseline on ImageNet classification with 12% fewer FLOPS and 29% fewer parameters. On COCO object detection, a pure self-attention model matches the mAP of a baseline RetinaNet while having 39% fewer FLOPS and 34% fewer parameters. Detailed ablation studies demonstrate that self-attention is especially impactful when used in later layers. These results establish that stand-alone self-attention is an important addition to the vision practitioner's toolbox.
    * Comments:
        * > Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019, Wang et al., 2020). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. (ViT, 2020)
        * > Wang et al. show that (single-headed) self-attention is a form of non-local means and that integrating it into a ResNet improves several tasks. Ramachandran et al. explore this direction further with stand-alone self-attention networks for vision. They report difficulties in designing an attention-based network stem and present a bespoke solution that avoids convolutions. (Early Convolutions Help Transformers See Better, 2021)
        * > (2021, PVT) Ramachandran et al. [35] proposed stand-alone self-attention was propose to replace convolutional layers with local self-attention units.
* [[Exploring Self-attention for Image Recognition](https://arxiv.org/abs/2004.13621)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2004.13621.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2004.13621/)]
    * Title: Exploring Self-attention for Image Recognition
    * Year: 28 Apr `2020`
    * Authors: Hengshuang Zhao, Jiaya Jia, Vladlen Koltun
    * Abstract: Recent work has shown that self-attention can serve as a basic building block for image recognition models. We explore variations of self-attention and assess their effectiveness for image recognition. We consider two forms of self-attention. One is pairwise self-attention, which generalizes standard dot-product attention and is fundamentally a set operator. The other is patchwise self-attention, which is strictly more powerful than convolution. Our pairwise self-attention networks match or outperform their convolutional counterparts, and the patchwise models substantially outperform the convolutional baselines. We also conduct experiments that probe the robustness of learned representations and conclude that self-attention networks may have significant benefits in terms of robustness and generalization.
    * Comments:
        * > Zhao et al. explore a broader set of self-attention operations with hard-coded locality constraints, more similar to standard CNNs. (Early Convolutions Help Transformers See Better, 2021)

## Improving self-attention (4 + 3 + 2)

> Several methods have been proposed for optimizing the self-attention operation in transformers (not necessarily for ViTs). Among these, a widely studied approach in sequence modeling tasks is to introduce sparsity in self-attention layers, wherein each token attends to a subset of tokens in an input sequence. Though these approaches reduces the time complexity from $O(k^{2})$ to $O(k\sqrt{k})$ or $O(k\log{k})$, the cost is a performance drop. Another popular approach for approximating self-attention is via low-rank approximation. Linformer decomposes the self-attention operation into multiple smaller self-attention operations via linear projections, and reduces the complexity of self-attention from $O(k^{2})$ to $O(k)$. However, Linformer still uses costly operations (e.g., batch-wise matrix multiplication for learning global representations in MHA, which may hinder the deployment of these models on resource-constrained devices. (MobileViTv2, 2022)

> The first line of research introduces locality to address the computational bottleneck in MHA. Instead of attending to all $k$ tokens, these methods use predefined patterns to limit the receptive field of self-attention from all $k$ tokens to a subset of tokens, reducing the time complexity from $O(k^{2})$ to $O(k\sqrt{k})$ or $O(k\log{k})$. However, such methods suffer from large performance degradation with moderate training/inference speed-up over standard MHA in transformers. (MobileViTv2, 2022)

* [[Sparse Transformers](https://arxiv.org/abs/1904.10509)]
    [[pdf](https://arxiv.org/pdf/1904.10509.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.10509/)]
    * Title: Generating Long Sequences with Sparse Transformers
    * Year: 23 Apr `2019`
    * Authors: Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever
    * Abstract: Transformers are powerful sequence models, but require time and memory that grows quadratically with the sequence length. In this paper we introduce sparse factorizations of the attention matrix which reduce this to $O(n \sqrt{n})$. We also introduce a) a variation on architecture and initialization to train deeper networks, b) the recomputation of attention matrices to save memory, and c) fast attention kernels for training. We call networks with these changes Sparse Transformers, and show they can model sequences tens of thousands of timesteps long using hundreds of layers. We use the same architecture to model images, audio, and text from raw bytes, setting a new state of the art for density modeling of Enwik8, CIFAR-10, and ImageNet-64. We generate unconditional samples that demonstrate global coherence and great diversity, and show it is possible in principle to use self-attention to model sequences of length one million or more.
    * Comments:
        * > In a different line of work, Sparse Transformers (Child et al., 2019) employ scalable approximations to global self-attention in order to be applicable to images. (ViT, )
* [[Longformer](https://arxiv.org/abs/2004.05150)]
    [[pdf](https://arxiv.org/pdf/2004.05150.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2004.05150/)]
    * Title: Longformer: The Long-Document Transformer
    * Year: 10 Apr `2020`
    * Authors: Iz Beltagy, Matthew E. Peters, Arman Cohan
    * Abstract: Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization dataset.
* [[Image Transformer](https://arxiv.org/abs/1802.05751)]
    [[pdf](https://arxiv.org/pdf/1802.05751.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1802.05751/)]
    * Title: Image Transformer
    * Year: 15 Feb `2018`
    * Authors: Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Łukasz Kaiser, Noam Shazeer, Alexander Ku, Dustin Tran
    * Abstract: Image generation has been successfully cast as an autoregressive sequence generation or transformation problem. Recent work has shown that self-attention is an effective way of modeling textual sequences. In this work, we generalize a recently proposed model architecture based on self-attention, the Transformer, to a sequence modeling formulation of image generation with a tractable likelihood. By restricting the self-attention mechanism to attend to local neighborhoods we significantly increase the size of images the model can process in practice, despite maintaining significantly larger receptive fields per layer than typical convolutional neural networks. While conceptually simple, our generative models significantly outperform the current state of the art in image generation on ImageNet, improving the best published negative log-likelihood on ImageNet from 3.83 to 3.77. We also present results on image super-resolution with a large magnification ratio, applying an encoder-decoder configuration of our architecture. In a human evaluation study, we find that images generated by our super-resolution model fool human observers three times more often than the previous state of the art.
    * Comments:
        * > (2019, AANet) Multiple positional encodings that augment activation maps with explicit spatial information have been proposed to alleviate related issues. In particular, the Image Transformer [32] extends the sinusoidal waves first introduced in the original Transformer [43] to 2 dimensional inputs and CoordConv [29] concatenates positional channels to an activation map.
* [[Blockwise Self-Attention for Long Document Understanding](https://arxiv.org/abs/1911.02972)]
    [[pdf](https://arxiv.org/pdf/1911.02972.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1911.02972/)]
    * Title: Blockwise Self-Attention for Long Document Understanding
    * Year: 07 Nov `2019`
    * Authors: Jiezhong Qiu, Hao Ma, Omer Levy, Scott Wen-tau Yih, Sinong Wang, Jie Tang
    * Abstract: We present BlockBERT, a lightweight and efficient BERT model for better modeling long-distance dependencies. Our model extends BERT by introducing sparse block structures into the attention matrix to reduce both memory consumption and training/inference time, which also enables attention heads to capture either short- or long-range contextual information. We conduct experiments on language model pre-training and several benchmark question answering datasets with various paragraph lengths. BlockBERT uses 18.7-36.1% less memory and 12.0-25.1% less time to learn the model. During testing, BlockBERT saves 27.8% inference time, while having comparable and sometimes better prediction accuracy, compared to an advanced BERT-based model, RoBERTa.

> To improve the efficiency of MHA, the second line of research uses similarity measures to group tokens. For instance, Reformer uses locality-sensitive hashing to group the tokens and reduces the theoretical self-attention cost from $O(k^{2})$ to $O(k\log{k})$. However, the efficiency gains over standard MHA are noticeable only for large sequences ($k > 2048$). Because $k < 1024$ in ViT, these approaches are not suitable for ViTs. (MobileViTv2, 2022)

* [[Reformer](https://arxiv.org/abs/2001.04451)]
    [[pdf](https://arxiv.org/pdf/2001.04451.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2001.04451/)]
    * Title: Reformer: The Efficient Transformer
    * Year: 13 Jan `2020`
    * Authors: Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya
    * Abstract: Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O($L^2$) to O($L\log L$), where $L$ is the length of the sequence. Furthermore, we use reversible residual layers instead of the standard residuals, which allows storing activations only once in the training process instead of $N$ times, where $N$ is the number of layers. The resulting model, the Reformer, performs on par with Transformer models while being much more memory-efficient and much faster on long sequences.
* [Fast transformers with clustered attention](https://dl.acm.org/doi/abs/10.5555/3495724.3497542)
    * Title: Fast transformers with clustered attention
    * Year: 15 April `2022`
    * Authors: Apoorv Vyas, Angelos Katharopoulos, François Fleuret
    * Abstract: Transformers have been proven a successful model for a variety of tasks in sequence modeling. However, computing the attention matrix, which is their key component, has quadratic complexity with respect to the sequence length, thus making them prohibitively expensive for large sequences. To address this, we propose clustered attention, which instead of computing the attention for every query, groups queries into clusters and computes attention just for the centroids. To further improve this approximation, we use the computed clusters to identify the keys with the highest attention per query and compute the exact key/query dot products. This results in a model with linear complexity with respect to the sequence length for a fixed number of clusters. We evaluate our approach on two automatic speech recognition datasets and show that our model consistently outperforms vanilla transformers for a given computational budget. Finally, we demonstrate that our model can approximate arbitrarily complex attention distributions with a minimal number of clusters by approximating a pretrained BERT model on GLUE and SQuAD benchmarks with only 25 clusters and no loss in performance.
* [Cluster-Former: Clustering-based Sparse Transformer for Question Answering](https://aclanthology.org/2021.findings-acl.346/)
    * Title: Cluster-Former: Clustering-based Sparse Transformer for Question Answering
    * Year: August `2021`
    * Author: Shuohang Wang, Luowei Zhou, Zhe Gan, Yen-Chun Chen, Yuwei Fang, Siqi Sun, Yu Cheng, Jingjing Liu
    * Abstract: Transformer has become ubiquitous in the deep learning field. One of the key ingredients that destined its success is the self-attention mechanism, which allows fully-connected contextual encoding over input tokens. However, despite its effectiveness in modeling short sequences, self-attention suffers when handling inputs with extreme long-range dependencies, as its complexity grows quadratically w.r.t. the sequence length. Therefore, long sequences are often encoded by Transformer in chunks using a sliding window. In this paper, we propose Cluster-Former, a novel clustering based sparse Transformer to perform attention across chunked sequences. The proposed framework is pivoted on two unique types of Transformer layer: Sliding-Window Layer and Cluster-Former Layer, which encode local sequence information and global context jointly and iteratively. This new design allows information integration beyond local windows, which is especially beneficial for question answering (QA) tasks that rely on long-range dependencies. Experiments show that ClusterFormer achieves state-of-the-art performance on several major QA benchmarks.

> The third line of research improves the efficiency of MHA via low-rank approximation. The main idea is to approximate the self-attention matrix with a low-rank matrix, reducing the computational cost from $O(k^{2})$ to $O(k)$. Even though these methods speed-up the self-attention operation significantly, they still use expensive operations for computing attention, which may hinder the deployment of these models on resource-constrained devices. (MobileViTv2, 2022)

* [[Linformer](https://arxiv.org/abs/2006.04768)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2006.04768.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2006.04768/)]
    * Title: Linformer: Self-Attention with Linear Complexity
    * Year: 08 Jun `2020`
    * Authors: Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
    * Abstract: Large transformer models have shown extraordinary success in achieving state-of-the-art results in many natural language processing applications. However, training and deploying these models can be prohibitively costly for long sequences, as the standard self-attention mechanism of the Transformer uses $O(n^2)$ time and space with respect to sequence length. In this paper, we demonstrate that the self-attention mechanism can be approximated by a low-rank matrix. We further exploit this finding to propose a new self-attention mechanism, which reduces the overall self-attention complexity from $O(n^2)$ to $O(n)$ in both time and space. The resulting linear transformer, the \textit{Linformer}, performs on par with standard Transformer models, while being much more memory- and time-efficient.
* [[Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)]
    [[pdf](https://arxiv.org/pdf/2009.14794.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2009.14794/)]
    * Title: Rethinking Attention with Performers
    * Year: 30 Sep `2020`
    * Authors: Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller
    * Abstract: We introduce Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attention-kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can be also used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. We demonstrate competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.

## Scaling up of vision models (3 + 5 + 3)

> An alternative way to scale attention is to apply it in blocks of varying sizes (Weissenborn et al., 2019), in the extreme case only along individual axes (Ho et al., 2019; Wang et al., 2020a). (ViT, )

* [[Scaling Autoregressive Video Models](https://arxiv.org/abs/1906.02634)]
    [[pdf](https://arxiv.org/pdf/1906.02634.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1906.02634/)]
    * Title: Scaling Autoregressive Video Models
    * Year: 06 Jun `2019`
    * Authors: Dirk Weissenborn, Oscar Täckström, Jakob Uszkoreit
    * Abstract: Due to the statistical complexity of video, the high degree of inherent stochasticity, and the sheer amount of data, generating natural video remains a challenging task. State-of-the-art video generation models often attempt to address these issues by combining sometimes complex, usually video-specific neural network architectures, latent variable models, adversarial training and a range of other methods. Despite their often high complexity, these approaches still fall short of generating high quality video continuations outside of narrow domains and often struggle with fidelity. In contrast, we show that conceptually simple autoregressive video generation models based on a three-dimensional self-attention mechanism achieve competitive results across multiple metrics on popular benchmark datasets, for which they produce continuations of high fidelity and realism. We also present results from training our models on Kinetics, a large scale action recognition dataset comprised of YouTube videos exhibiting phenomena such as camera movement, complex object interactions and diverse human movement. While modeling these phenomena consistently remains elusive, we hope that our results, which include occasional realistic continuations encourage further research on comparatively complex, large scale datasets such as Kinetics.
* [[Axial Attention in Multidimensional Transformers](https://arxiv.org/abs/1912.12180)]
    [[pdf](https://arxiv.org/pdf/1912.12180.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1912.12180/)]
    * Title: Axial Attention in Multidimensional Transformers
    * Year: 20 Dec `2019`
    * Authors: Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, Tim Salimans
    * Abstract: We propose Axial Transformers, a self-attention-based autoregressive model for images and other data organized as high dimensional tensors. Existing autoregressive models either suffer from excessively large computational resource requirements for high dimensional data, or make compromises in terms of distribution expressiveness or ease of implementation in order to decrease resource requirements. Our architecture, by contrast, maintains both full expressiveness over joint distributions over data and ease of implementation with standard deep learning frameworks, while requiring reasonable memory and computation and achieving state-of-the-art results on standard generative modeling benchmarks. Our models are based on axial attention, a simple generalization of self-attention that naturally aligns with the multiple dimensions of the tensors in both the encoding and the decoding settings. Notably the proposed structure of the layers allows for the vast majority of the context to be computed in parallel during decoding without introducing any independence assumptions. This semi-parallel structure goes a long way to making decoding from even a very large Axial Transformer broadly applicable. We demonstrate state-of-the-art results for the Axial Transformer on the ImageNet-32 and ImageNet-64 image benchmarks as well as on the BAIR Robotic Pushing video benchmark. We open source the implementation of Axial Transformers.
* [[Axial-DeepLab](https://arxiv.org/abs/2003.07853)]
    [[pdf](https://arxiv.org/pdf/2003.07853.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2003.07853/)]
    * Title: Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation
    * Year: 17 Mar `2020`
    * Authors: Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, Liang-Chieh Chen
    * Abstract: Convolution exploits locality for efficiency at a cost of missing long range context. Self-attention has been adopted to augment CNNs with non-local interactions. Recent works prove it possible to stack self-attention layers to obtain a fully attentional network by restricting the attention to a local region. In this paper, we attempt to remove this constraint by factorizing 2D self-attention into two 1D self-attentions. This reduces computation complexity and allows performing attention within a larger or even global region. In companion, we also propose a position-sensitive self-attention design. Combining both yields our position-sensitive axial-attention layer, a novel building block that one could stack to form axial-attention models for image classification and dense prediction. We demonstrate the effectiveness of our model on four large-scale datasets. In particular, our model outperforms all existing stand-alone self-attention models on ImageNet. Our Axial-DeepLab improves 2.8% PQ over bottom-up state-of-the-art on COCO test-dev. This previous state-of-the-art is attained by our small variant that is 3.8x parameter-efficient and 27x computation-efficient. Axial-DeepLab also achieves state-of-the-art results on Mapillary Vistas and Cityscapes.
    * Comments:
        * > Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019, Wang et al., 2020). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. (ViT, 2020)

> On the other hand, the scaling up of vision models has been lagging behind. While it has long been recognized that larger vision models usually perform better on vision tasks, the absolute model size was just able to reach about 1-2 billion parameters very recently. (Swin Transformer V2, 2021)

* [[CoAtNet](https://arxiv.org/abs/2106.04803)]
    [[pdf](https://arxiv.org/pdf/2106.04803.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.04803/)]
    * Title: CoAtNet: Marrying Convolution and Attention for All Data Sizes
    * Year: 09 Jun `2021`
    * Authors: Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan
    * Abstract: Transformers have attracted increasing interests in computer vision, but they still fall behind state-of-the-art convolutional networks. In this work, we show that while Transformers tend to have larger model capacity, their generalization can be worse than convolutional networks due to the lack of the right inductive bias. To effectively combine the strengths from both architectures, we present CoAtNets(pronounced "coat" nets), a family of hybrid models built from two key insights: (1) depthwise Convolution and self-Attention can be naturally unified via simple relative attention; (2) vertically stacking convolution layers and attention layers in a principled way is surprisingly effective in improving generalization, capacity and efficiency. Experiments show that our CoAtNets achieve state-of-the-art performance under different resource constraints across various datasets: Without extra data, CoAtNet achieves 86.0% ImageNet top-1 accuracy; When pre-trained with 13M images from ImageNet-21K, our CoAtNet achieves 88.56% top-1 accuracy, matching ViT-huge pre-trained with 300M images from JFT-300M while using 23x less data; Notably, when we further scale up CoAtNet with JFT-3B, it achieves 90.88% top-1 accuracy on ImageNet, establishing a new state-of-the-art result.
* [[Self-supervised Pretraining of Visual Features in the Wild](https://arxiv.org/abs/2103.01988)]
    [[pdf](https://arxiv.org/pdf/2103.01988.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.01988/)]
    * Title: Self-supervised Pretraining of Visual Features in the Wild
    * Year: 02 Mar `2021`
    * Authors: Priya Goyal, Mathilde Caron, Benjamin Lefaudeux, Min Xu, Pengchao Wang, Vivek Pai, Mannat Singh, Vitaliy Liptchinsky, Ishan Misra, Armand Joulin, Piotr Bojanowski
    * Abstract: Recently, self-supervised learning methods like MoCo, SimCLR, BYOL and SwAV have reduced the gap with supervised methods. These results have been achieved in a control environment, that is the highly curated ImageNet dataset. However, the premise of self-supervised learning is that it can learn from any random image and from any unbounded dataset. In this work, we explore if self-supervision lives to its expectation by training large models on random, uncurated images with no supervision. Our final SElf-supERvised (SEER) model, a RegNetY with 1.3B parameters trained on 1B random images with 512 GPUs achieves 84.2% top-1 accuracy, surpassing the best self-supervised pretrained model by 1% and confirming that self-supervised learning works in a real world setting. Interestingly, we also observe that self-supervised models are good few-shot learners achieving 77.9% top-1 with access to only 10% of ImageNet. Code: this https URL
* Big Transfer (BiT): General Visual Representation Learning
* [Scaling Vision with Sparse Mixture of Experts](https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html)
    * Title: Scaling Vision with Sparse Mixture of Experts
    * Year: `2021`
    * Author: Carlos Riquelme
    * Abstract: Sparsely-gated Mixture of Experts networks (MoEs) have demonstrated excellent scalability in Natural Language Processing. In Computer Vision, however, almost all performant networks are "dense", that is, every input is processed by every parameter. We present a Vision MoE (V-MoE), a sparse version of the Vision Transformer, that is scalable and competitive with the largest dense networks. When applied to image recognition, V-MoE matches the performance of state-of-the-art networks, while requiring as little as half of the compute at inference time. Further, we propose an extension to the routing algorithm that can prioritize subsets of each input across the entire batch, leading to adaptive per-image compute. This allows V-MoE to trade-off performance and compute smoothly at test-time. Finally, we demonstrate the potential of V-MoE to scale vision models, and train a 15B parameter model that attains 90.35% on ImageNet.
* [[Scaling Vision Transformers](https://arxiv.org/abs/2106.04560)]
    [[pdf](https://arxiv.org/pdf/2106.04560.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.04560/)]
    * Title: Scaling Vision Transformers
    * Year: 08 Jun `2021`
    * Authors: Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, Lucas Beyer
    * Abstract: Attention-based neural networks such as the Vision Transformer (ViT) have recently attained state-of-the-art results on many computer vision benchmarks. Scale is a primary ingredient in attaining excellent results, therefore, understanding a model's scaling properties is a key to designing future generations effectively. While the laws for scaling Transformer language models have been studied, it is unknown how Vision Transformers scale. To address this, we scale ViT models and data, both up and down, and characterize the relationships between error rate, data, and compute. Along the way, we refine the architecture and training of ViT, reducing memory consumption and increasing accuracy of the resulting models. As a result, we successfully train a ViT model with two billion parameters, which attains a new state-of-the-art on ImageNet of 90.45% top-1 accuracy. The model also performs well for few-shot transfer, for example, reaching 84.86% top-1 accuracy on ImageNet with only 10 examples per class.

> The general trend is to increase the number of parameters in ViT networks to improve the performance. (MobileViTv1, 2021)

* Training data-efficient image transformers & distillation through attention
* [[LeViT](https://arxiv.org/abs/2104.01136)]
    [[pdf](https://arxiv.org/pdf/2104.01136.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2104.01136/)]
    * Title: LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference
    * Year: 02 Apr `2021`
    * Authors: Ben Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze
    * Abstract: We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime. Our work exploits recent findings in attention-based architectures, which are competitive on highly parallel processing hardware. We revisit principles from the extensive literature on convolutional neural networks to apply them to transformers, in particular activation maps with decreasing resolutions. We also introduce the attention bias, a new way to integrate positional information in vision transformers. As a result, we propose LeVIT: a hybrid neural network for fast inference image classification. We consider different measures of efficiency on different hardware platforms, so as to best reflect a wide range of application scenarios. Our extensive experiments empirically validate our technical choices and show they are suitable to most architectures. Overall, LeViT significantly outperforms existing convnets and vision transformers with respect to the speed/accuracy tradeoff. For example, at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU. We release the code at this https URL
* [[CvT](https://arxiv.org/abs/2103.15808)]
    [[pdf](https://arxiv.org/pdf/2103.15808.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.15808/)]
    * Title: CvT: Introducing Convolutions to Vision Transformers
    * Year: 29 Mar `2021`
    * Authors: Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang
    * Abstract: We present in this paper a new architecture, named Convolutional vision Transformer (CvT), that improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs. This is accomplished through two primary modifications: a hierarchy of Transformers containing a new convolutional token embedding, and a convolutional Transformer block leveraging a convolutional projection. These changes introduce desirable properties of convolutional neural networks (CNNs) to the ViT architecture (\ie shift, scale, and distortion invariance) while maintaining the merits of Transformers (\ie dynamic attention, global context, and better generalization). We validate CvT by conducting extensive experiments, showing that this approach achieves state-of-the-art performance over other Vision Transformers and ResNets on ImageNet-1k, with fewer parameters and lower FLOPs. In addition, performance gains are maintained when pretrained on larger datasets (\eg ImageNet-22k) and fine-tuned to downstream tasks. Pre-trained on ImageNet-22k, our CvT-W24 obtains a top-1 accuracy of 87.7\% on the ImageNet-1k val set. Finally, our results show that the positional encoding, a crucial component in existing Vision Transformers, can be safely removed in our model, simplifying the design for higher resolution vision tasks. Code will be released at \url{this https URL}.

## ResNet-Like Architectures (2020, ViT) (count=3)

* [[Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/abs/1805.00932)]
    [[pdf](https://arxiv.org/pdf/1805.00932.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1805.00932/)]
    * Title: Exploring the Limits of Weakly Supervised Pretraining
    * Year: 02 May `2018`
    * Authors: Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, Laurens van der Maaten
    * Abstract: State-of-the-art visual perception models for a wide range of tasks rely on supervised pretraining. ImageNet classification is the de facto pretraining task for these models. Yet, ImageNet is now nearly ten years old and is by modern standards "small". Even so, relatively little is known about the behavior of pretraining with datasets that are multiple orders of magnitude larger. The reasons are obvious: such datasets are difficult to collect and annotate. In this paper, we present a unique study of transfer learning with large convolutional networks trained to predict hashtags on billions of social media images. Our experiments demonstrate that training for large-scale hashtag prediction leads to excellent results. We show improvements on several image classification and object detection tasks, and report the highest ImageNet-1k single-crop, top-1 accuracy to date: 85.4% (97.6% top-5). We also perform extensive experiments that provide novel empirical data on the relationship between large-scale pretraining and transfer learning performance.
* [[Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252)]
    [[pdf](https://arxiv.org/pdf/1911.04252.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1911.04252/)]
    * Title: Self-training with Noisy Student improves ImageNet classification
    * Year: 11 Nov `2019`
    * Authors: Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le
    * Abstract: We present Noisy Student Training, a semi-supervised learning approach that works well even when labeled data is abundant. Noisy Student Training achieves 88.4% top-1 accuracy on ImageNet, which is 2.0% better than the state-of-the-art model that requires 3.5B weakly labeled Instagram images. On robustness test sets, it improves ImageNet-A top-1 accuracy from 61.0% to 83.7%, reduces ImageNet-C mean corruption error from 45.7 to 28.3, and reduces ImageNet-P mean flip rate from 27.8 to 12.2. Noisy Student Training extends the idea of self-training and distillation with the use of equal-or-larger student models and noise added to the student during learning. On ImageNet, we first train an EfficientNet model on labeled images and use it as a teacher to generate pseudo labels for 300M unlabeled images. We then train a larger EfficientNet as a student model on the combination of labeled and pseudo labeled images. We iterate this process by putting back the student as the teacher. During the learning of the student, we inject noise such as dropout, stochastic depth, and data augmentation via RandAugment to the student so that the student generalizes better than the teacher. Models are available at this https URL. Code is available at this https URL.
* [[Big Transfer (BiT)](https://arxiv.org/abs/1912.11370)]
    [[pdf](https://arxiv.org/pdf/1912.11370.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1912.11370/)]
    * Title: Big Transfer (BiT): General Visual Representation Learning
    * Year: 24 Dec `2019`
    * Authors: Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby
    * Abstract: Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By combining a few carefully selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets. BiT performs well across a surprisingly wide range of data regimes -- from 1 example per class to 1M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis of the main components that lead to high transfer performance.

## Vision Transformer Variants at Relatively Small Scale (11)

* [[Twins](https://arxiv.org/abs/2104.13840)]
    [[pdf](https://arxiv.org/pdf/2104.13840.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2104.13840/)]
    * Title: Twins: Revisiting the Design of Spatial Attention in Vision Transformers
    * Year: 28 Apr `2021`
    * Authors: Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, Chunhua Shen
    * Abstract: Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully-devised yet simple spatial attention mechanism performs favourably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins-PCPVT and Twins-SVT. Our proposed architectures are highly-efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks, including image level classification as well as dense detection and segmentation. The simplicity and strong performance suggest that our proposed architectures may serve as stronger backbones for many vision tasks. Our code is released at this https URL .
* [[CSWin Transformer](https://arxiv.org/abs/2107.00652)]
    [[pdf](https://arxiv.org/pdf/2107.00652.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2107.00652/)]
    * Title: CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows
    * Year: 01 Jul `2021`
    * Authors: Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, Baining Guo
    * Abstract: We present CSWin Transformer, an efficient and effective Transformer-based backbone for general-purpose vision tasks. A challenging issue in Transformer design is that global self-attention is very expensive to compute whereas local self-attention often limits the field of interactions of each token. To address this issue, we develop the Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel that form a cross-shaped window, with each stripe obtained by splitting the input feature into stripes of equal width. We provide a mathematical analysis of the effect of the stripe width and vary the stripe width for different layers of the Transformer network which achieves strong modeling capability while limiting the computation cost. We also introduce Locally-enhanced Positional Encoding (LePE), which handles the local positional information better than existing encoding schemes. LePE naturally supports arbitrary input resolutions, and is thus especially effective and friendly for downstream tasks. Incorporated with these designs and a hierarchical structure, CSWin Transformer demonstrates competitive performance on common vision tasks. Specifically, it achieves 85.4\% Top-1 accuracy on ImageNet-1K without any extra training data or label, 53.9 box AP and 46.4 mask AP on the COCO detection task, and 52.2 mIOU on the ADE20K semantic segmentation task, surpassing previous state-of-the-art Swin Transformer backbone by +1.2, +2.0, +1.4, and +2.0 respectively under the similar FLOPs setting. By further pretraining on the larger dataset ImageNet-21K, we achieve 87.5% Top-1 accuracy on ImageNet-1K and high segmentation performance on ADE20K with 55.7 mIoU. The code and models are available at this https URL.
* [[Shuffle Transformer](https://arxiv.org/abs/2106.03650)]
    [[pdf](https://arxiv.org/pdf/2106.03650.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.03650/)]
    * Title: Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer
    * Year: 07 Jun `2021`
    * Authors: Zilong Huang, Youcheng Ben, Guozhong Luo, Pei Cheng, Gang Yu, Bin Fu
    * Abstract: Very recently, Window-based Transformers, which computed self-attention within non-overlapping local windows, demonstrated promising results on image classification, semantic segmentation, and object detection. However, less study has been devoted to the cross-window connection which is the key element to improve the representation ability. In this work, we revisit the spatial shuffle as an efficient way to build connections among windows. As a result, we propose a new vision transformer, named Shuffle Transformer, which is highly efficient and easy to implement by modifying two lines of code. Furthermore, the depth-wise convolution is introduced to complement the spatial shuffle for enhancing neighbor-window connections. The proposed architectures achieve excellent performance on a wide range of visual tasks including image-level classification, object detection, and semantic segmentation. Code will be released for reproduction.
* [[LocalViT](https://arxiv.org/abs/2104.05707)]
    [[pdf](https://arxiv.org/pdf/2104.05707.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2104.05707/)]
    * Title: LocalViT: Bringing Locality to Vision Transformers
    * Year: 12 Apr `2021`
    * Authors: Yawei Li, Kai Zhang, Jiezhang Cao, Radu Timofte, Luc Van Gool
    * Abstract: We study how to introduce locality mechanisms into vision transformers. The transformer network originates from machine translation and is particularly good at modelling long-range dependencies within a long sequence. Although the global interaction between the token embeddings could be well modelled by the self-attention mechanism of transformers, what is lacking a locality mechanism for information exchange within a local region. Yet, locality is essential for images since it pertains to structures like lines, edges, shapes, and even objects. We add locality to vision transformers by introducing depth-wise convolution into the feed-forward network. This seemingly simple solution is inspired by the comparison between feed-forward networks and inverted residual blocks. The importance of locality mechanisms is validated in two ways: 1) A wide range of design choices (activation function, layer placement, expansion ratio) are available for incorporating locality mechanisms and all proper choices can lead to a performance gain over the baseline, and 2) The same locality mechanism is successfully applied to 4 vision transformers, which shows the generalization of the locality concept. In particular, for ImageNet2012 classification, the locality-enhanced transformers outperform the baselines DeiT-T and PVT-T by 2.6\% and 3.1\% with a negligible increase in the number of parameters and computational effort. Code is available at \url{this https URL}.
* DeiT
* Pyramid Vision Transformer
* [[Early Convolutions Help Transformers See Better](https://arxiv.org/abs/2106.14881)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2106.14881.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.14881/)]
    * Title: Early Convolutions Help Transformers See Better
    * Year: 28 Jun `2021`
    * Authors: Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Dollár, Ross Girshick
    * Abstract: Vision transformer (ViT) models exhibit substandard optimizability. In particular, they are sensitive to the choice of optimizer (AdamW vs. SGD), optimizer hyperparameters, and training schedule length. In comparison, modern convolutional neural networks are easier to optimize. Why is this the case? In this work, we conjecture that the issue lies with the patchify stem of ViT models, which is implemented by a stride-p p*p convolution (p=16 by default) applied to the input image. This large-kernel plus large-stride convolution runs counter to typical design choices of convolutional layers in neural networks. To test whether this atypical design choice causes an issue, we analyze the optimization behavior of ViT models with their original patchify stem versus a simple counterpart where we replace the ViT stem by a small number of stacked stride-two 3*3 convolutions. While the vast majority of computation in the two ViT designs is identical, we find that this small change in early visual processing results in markedly different training behavior in terms of the sensitivity to optimization settings as well as the final model accuracy. Using a convolutional stem in ViT dramatically increases optimization stability and also improves peak performance (by ~1-2% top-1 accuracy on ImageNet-1k), while maintaining flops and runtime. The improvement can be observed across the wide spectrum of model complexities (from 1G to 36G flops) and dataset scales (from ImageNet-1k to ImageNet-21k). These findings lead us to recommend using a standard, lightweight convolutional stem for ViT models in this regime as a more robust architectural choice compared to the original ViT model design.
* [[Focal Transformer](https://arxiv.org/abs/2107.00641)]
    [[pdf](https://arxiv.org/pdf/2107.00641.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2107.00641/)]
    * Title: Focal Self-attention for Local-Global Interactions in Vision Transformers
    * Year: 01 Jul `2021`
    * Authors: Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, Jianfeng Gao
    * Abstract: Recently, Vision Transformer and its variants have shown great promise on various computer vision tasks. The ability of capturing short- and long-range visual dependencies through self-attention is arguably the main source for the success. But it also brings challenges due to quadratic computational overhead, especially for the high-resolution vision tasks (e.g., object detection). In this paper, we present focal self-attention, a new mechanism that incorporates both fine-grained local and coarse-grained global interactions. Using this new mechanism, each token attends the closest surrounding tokens at fine granularity but the tokens far away at coarse granularity, and thus can capture both short- and long-range visual dependencies efficiently and effectively. With focal self-attention, we propose a new variant of Vision Transformer models, called Focal Transformer, which achieves superior performance over the state-of-the-art vision Transformers on a range of public image classification and object detection benchmarks. In particular, our Focal Transformer models with a moderate size of 51.1M and a larger size of 89.8M achieve 83.5 and 83.8 Top-1 accuracy, respectively, on ImageNet classification at 224x224 resolution. Using Focal Transformers as the backbones, we obtain consistent and substantial improvements over the current state-of-the-art Swin Transformers for 6 different object detection methods trained with standard 1x and 3x schedules. Our largest Focal Transformer yields 58.7/58.9 box mAPs and 50.9/51.3 mask mAPs on COCO mini-val/test-dev, and 55.4 mIoU on ADE20K for semantic segmentation, creating new SoTA on three of the most challenging computer vision tasks.
* Tokens-to-Token ViT
* [[VOLO](https://arxiv.org/abs/2106.13112)]
    [[pdf](https://arxiv.org/pdf/2106.13112.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.13112/)]
    * Title: VOLO: Vision Outlooker for Visual Recognition
    * Year: 24 Jun `2021`
    * Authors: Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, Shuicheng Yan
    * Abstract: Visual recognition has been dominated by convolutional neural networks (CNNs) for years. Though recently the prevailing vision transformers (ViTs) have shown great potential of self-attention based models in ImageNet classification, their performance is still inferior to that of the latest SOTA CNNs if no extra data are provided. In this work, we try to close the performance gap and demonstrate that attention-based models are indeed able to outperform CNNs. We find a major factor limiting the performance of ViTs for ImageNet classification is their low efficacy in encoding fine-level features into the token representations. To resolve this, we introduce a novel outlook attention and present a simple and general architecture, termed Vision Outlooker (VOLO). Unlike self-attention that focuses on global dependency modeling at a coarse level, the outlook attention efficiently encodes finer-level features and contexts into tokens, which is shown to be critically beneficial to recognition performance but largely ignored by the self-attention. Experiments show that our VOLO achieves 87.1% top-1 accuracy on ImageNet-1K classification, which is the first model exceeding 87% accuracy on this competitive benchmark, without using any extra training data In addition, the pre-trained VOLO transfers well to downstream tasks, such as semantic segmentation. We achieve 84.3% mIoU score on the cityscapes validation set and 54.3% on the ADE20K validation set. Code is available at \url{this https URL}.
* [[Multi-Scale Vision Longformer](https://arxiv.org/abs/2103.15358)]
    [[pdf](https://arxiv.org/pdf/2103.15358.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.15358/)]
    * Title: Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding
    * Year: 29 Mar `2021`
    * Authors: Pengchuan Zhang, Xiyang Dai, Jianwei Yang, Bin Xiao, Lu Yuan, Lei Zhang, Jianfeng Gao
    * Abstract: This paper presents a new Vision Transformer (ViT) architecture Multi-Scale Vision Longformer, which significantly enhances the ViT of \cite{dosovitskiy2020image} for encoding high-resolution images using two techniques. The first is the multi-scale model structure, which provides image encodings at multiple scales with manageable computational cost. The second is the attention mechanism of vision Longformer, which is a variant of Longformer \cite{beltagy2020longformer}, originally developed for natural language processing, and achieves a linear complexity w.r.t. the number of input tokens. A comprehensive empirical study shows that the new ViT significantly outperforms several strong baselines, including the existing ViT models and their ResNet counterparts, and the Pyramid Vision Transformer from a concurrent work \cite{wang2021pyramid}, on a range of vision tasks, including image classification, object detection, and segmentation. The models and source code are released at \url{this https URL}.

## Multi-scale networks (5)

* [[Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227)]
    [[pdf](https://arxiv.org/pdf/2104.11227.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2104.11227/)]
    * Title: Multiscale Vision Transformers
    * Year: 22 Apr `2021`
    * Authors: Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    * Abstract: We present Multiscale Vision Transformers (MViT) for video and image recognition, by connecting the seminal idea of multiscale feature hierarchies with transformer models. Multiscale Transformers have several channel-resolution scale stages. Starting from the input resolution and a small channel dimension, the stages hierarchically expand the channel capacity while reducing the spatial resolution. This creates a multiscale pyramid of features with early layers operating at high spatial resolution to model simple low-level visual information, and deeper layers at spatially coarse, but complex, high-dimensional features. We evaluate this fundamental architectural prior for modeling the dense nature of visual signals for a variety of video recognition tasks where it outperforms concurrent vision transformers that rely on large scale external pre-training and are 5-10x more costly in computation and parameters. We further remove the temporal dimension and apply our model for image classification where it outperforms prior work on vision transformers. Code is available at: this https URL
* LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference
* Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
* Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
* Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet

## Locality Prior (5)

* [[Visformer](https://arxiv.org/abs/2104.12533)]
    [[pdf](https://arxiv.org/pdf/2104.12533.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2104.12533/)]
    * Title: Visformer: The Vision-friendly Transformer
    * Year: 26 Apr `2021`
    * Authors: Zhengsu Chen, Lingxi Xie, Jianwei Niu, Xuefeng Liu, Longhui Wei, Qi Tian
    * Abstract: The past year has witnessed the rapid development of applying the Transformer module to vision problems. While some researchers have demonstrated that Transformer-based models enjoy a favorable ability of fitting data, there are still growing number of evidences showing that these models suffer over-fitting especially when the training data is limited. This paper offers an empirical study by performing step-by-step operations to gradually transit a Transformer-based model to a convolution-based model. The results we obtain during the transition process deliver useful messages for improving visual recognition. Based on these observations, we propose a new architecture named Visformer, which is abbreviated from the `Vision-friendly Transformer'. With the same computational complexity, Visformer outperforms both the Transformer-based and convolution-based models in terms of ImageNet classification accuracy, and the advantage becomes more significant when the model complexity is lower or the training set is smaller. The code is available at this https URL.
* ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases
* LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference
* CvT: Introducing Convolutions to Vision Transformers
* [[Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/abs/2103.11816)]
    [[pdf](https://arxiv.org/pdf/2103.11816.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.11816/)]
    * Title: Incorporating Convolution Designs into Visual Transformers
    * Year: 22 Mar `2021`
    * Authors: Kun Yuan, Shaopeng Guo, Ziwei Liu, Aojun Zhou, Fengwei Yu, Wei Wu
    * Abstract: Motivated by the success of Transformers in natural language processing (NLP) tasks, there emerge some attempts (e.g., ViT and DeiT) to apply Transformers to the vision domain. However, pure Transformer architectures often require a large amount of training data or extra supervision to obtain comparable performance with convolutional neural networks (CNNs). To overcome these limitations, we analyze the potential drawbacks when directly borrowing Transformer architectures from NLP. Then we propose a new \textbf{Convolution-enhanced image Transformer (CeiT)} which combines the advantages of CNNs in extracting low-level features, strengthening locality, and the advantages of Transformers in establishing long-range dependencies. Three modifications are made to the original Transformer: \textbf{1)} instead of the straightforward tokenization from raw input images, we design an \textbf{Image-to-Tokens (I2T)} module that extracts patches from generated low-level features; \textbf{2)} the feed-froward network in each encoder block is replaced with a \textbf{Locally-enhanced Feed-Forward (LeFF)} layer that promotes the correlation among neighboring tokens in the spatial dimension; \textbf{3)} a \textbf{Layer-wise Class token Attention (LCA)} is attached at the top of the Transformer that utilizes the multi-level representations. Experimental results on ImageNet and seven downstream tasks show the effectiveness and generalization ability of CeiT compared with previous Transformers and state-of-the-art CNNs, without requiring a large amount of training data and extra CNN teachers. Besides, CeiT models also demonstrate better convergence with $3\times$ fewer training iterations, which can reduce the training cost significantly\footnote{Code and models will be released upon acceptance.}.

## Referenced by Swin Transformer V1

### (2021, Swin Transformer V1) Self-attention based backbone architectures (count=3)

> (2021, Swin Transformer V1) Also inspired by the success of self-attention layers and Transformer architectures in the NLP field, some works employ self-attention layers to replace some or all of the spatial convolution layers in the popular ResNet [32, 49, 77]. In these works, the self-attention is computed within a local window of each pixel to expedite optimization [32], and they achieve slightly better accuracy/FLOPs trade-offs than the counterpart ResNet architecture. However, their costly memory access causes their actual latency to be significantly larger than that of the convolutional networks [32]. Instead of using sliding windows, we propose to shift windows between consecutive layers, which allows for a more efficient implementation in general hardware.

* [[Local Relation Networks for Image Recognition](https://arxiv.org/abs/1904.11491)]
    [[pdf](https://arxiv.org/pdf/1904.11491.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.11491/)]
    * Title: Local Relation Networks for Image Recognition
    * Year: 25 Apr `2019`
    * Authors: Han Hu, Zheng Zhang, Zhenda Xie, Stephen Lin
    * Abstract: The convolution layer has been the dominant feature extractor in computer vision for years. However, the spatial aggregation in convolution is basically a pattern matching process that applies fixed filters which are inefficient at modeling visual elements with varying spatial distributions. This paper presents a new image feature extractor, called the local relation layer, that adaptively determines aggregation weights based on the compositional relationship of local pixel pairs. With this relational approach, it can composite visual elements into higher-level entities in a more efficient manner that benefits semantic inference. A network built with local relation layers, called the Local Relation Network (LR-Net), is found to provide greater modeling capacity than its counterpart built with regular convolution on large-scale recognition tasks such as ImageNet classification.
* Stand-Alone Self-Attention in Vision Models
* Exploring Self-attention for Image Recognition

### (2021, Swin Transformer V1) Self-attention/Transformers to complement CNN Backbones (count=6)

> (2021, Swin Transformer V1) Another line of work is to augment a standard CNN architecture with self-attention layers or Transformers. The self-attention layers can complement backbones [64, 6, 68, 22, 71, 54] or head networks [31, 26] by providing the capability to encode distant dependencies or heterogeneous interactions.

* Non-local Neural Networks
* [GCNet](https://ieeexplore.ieee.org/document/9022134)
    * Title: GCNet: Non-Local Networks Meet Squeeze-Excitation Networks and Beyond
    * Year: 05 March `2020`
    * Author: Yue Cao
    * Abstract: The Non-Local Network (NLNet) presents a pioneering approach for capturing long-range dependencies, via aggregating query-specific global context to each query position. However, through a rigorous empirical analysis, we have found that the global contexts modeled by non-local network are almost the same for different query positions within an image. In this paper, we take advantage of this finding to create a simplified network based on a query-independent formulation, which maintains the accuracy of NLNet but with significantly less computation. We further observe that this simplified design shares similar structure with Squeeze-Excitation Network (SENet). Hence we unify them into a three-step general framework for global context modeling. Within the general framework, we design a better instantiation, called the global context (GC) block, which is lightweight and can effectively model the global context. The lightweight property allows us to apply it for multiple layers in a backbone network to construct a global context network (GCNet), which generally outperforms both simplified NLNet and SENet on major benchmarks for various recognition tasks.
* [[Disentangled Non-Local Neural Networks](https://arxiv.org/abs/2006.06668)]
    [[pdf](https://arxiv.org/pdf/2006.06668.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2006.06668/)]
    * Title: Disentangled Non-Local Neural Networks
    * Year: 11 Jun `2020`
    * Authors: Minghao Yin, Zhuliang Yao, Yue Cao, Xiu Li, Zheng Zhang, Stephen Lin, Han Hu
    * Abstract: The non-local block is a popular module for strengthening the context modeling ability of a regular convolutional neural network. This paper first studies the non-local block in depth, where we find that its attention computation can be split into two terms, a whitened pairwise term accounting for the relationship between two pixels and a unary term representing the saliency of every pixel. We also observe that the two terms trained alone tend to model different visual clues, e.g. the whitened pairwise term learns within-region relationships while the unary term learns salient boundaries. However, the two terms are tightly coupled in the non-local block, which hinders the learning of each. Based on these findings, we present the disentangled non-local block, where the two terms are decoupled to facilitate learning for both terms. We demonstrate the effectiveness of the decoupled design on various tasks, such as semantic segmentation on Cityscapes, ADE20K and PASCAL Context, object detection on COCO, and action recognition on Kinetics.
* [[BoTNet](https://arxiv.org/abs/2101.11605)]
    [[pdf](https://arxiv.org/pdf/2101.11605.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2101.11605/)]
    * Title: Bottleneck Transformers for Visual Recognition
    * Year: 27 Jan `2021`
    * Authors: Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel, Ashish Vaswani
    * Abstract: We present BoTNet, a conceptually simple yet powerful backbone architecture that incorporates self-attention for multiple computer vision tasks including image classification, object detection and instance segmentation. By just replacing the spatial convolutions with global self-attention in the final three bottleneck blocks of a ResNet and no other changes, our approach improves upon the baselines significantly on instance segmentation and object detection while also reducing the parameters, with minimal overhead in latency. Through the design of BoTNet, we also point out how ResNet bottleneck blocks with self-attention can be viewed as Transformer blocks. Without any bells and whistles, BoTNet achieves 44.4% Mask AP and 49.7% Box AP on the COCO Instance Segmentation benchmark using the Mask R-CNN framework; surpassing the previous best published single model and single scale results of ResNeSt evaluated on the COCO validation set. Finally, we present a simple adaptation of the BoTNet design for image classification, resulting in models that achieve a strong performance of 84.7% top-1 accuracy on the ImageNet benchmark while being up to 1.64x faster in compute time than the popular EfficientNet models on TPU-v3 hardware. We hope our simple and effective approach will serve as a strong baseline for future research in self-attention models for vision
    * Comments:
        * > Lambda Networks (Bello, 2021) and BotNet (Srinivas et al., 2021) improve training speed by using attention layers in ConvNets. (EfficientNetV2, 2021)
        * > Lambda Networks (Bello, 2021), NFNets (Brock et al., 2021), BoTNets (Srinivas et al., 2021), ResNet-RS (Bello et al., 2021) focus on TPU training speed. (EfficientNetV2, 2021)

### (2021, Swin Transformer V1) Self-attention/Transformers to complement CNN heads (count=2)

> (2021, Swin Transformer V1) Another line of work is to augment a standard CNN architecture with self-attention layers or Transformers. The self-attention layers can complement backbones [64, 6, 68, 22, 71, 54] or head networks [31, 26] by providing the capability to encode distant dependencies or heterogeneous interactions.

* [[Relation Networks for Object Detection](https://arxiv.org/abs/1711.11575)]
    [[pdf](https://arxiv.org/pdf/1711.11575.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1711.11575/)]
    * Title: Relation Networks for Object Detection
    * Year: 30 Nov `2017`
    * Authors: Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, Yichen Wei
    * Abstract: Although it is well believed for years that modeling relations between objects would help object recognition, there has not been evidence that the idea is working in the deep learning era. All state-of-the-art object detection systems still rely on recognizing object instances individually, without exploiting their relations during learning. This work proposes an object relation module. It processes a set of objects simultaneously through interaction between their appearance feature and geometry, thus allowing modeling of their relations. It is lightweight and in-place. It does not require additional supervision and is easy to embed in existing networks. It is shown effective on improving object recognition and duplicate removal steps in the modern object detection pipeline. It verifies the efficacy of modeling object relations in CNN based detection. It gives rise to the first fully end-to-end object detector.
* [[Learning Region Features for Object Detection](https://arxiv.org/abs/1803.07066)]
    [[pdf](https://arxiv.org/pdf/1803.07066.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1803.07066/)]
    * Title: Learning Region Features for Object Detection
    * Year: 19 Mar `2018`
    * Authors: Jiayuan Gu, Han Hu, Liwei Wang, Yichen Wei, Jifeng Dai
    * Abstract: While most steps in the modern object detection methods are learnable, the region feature extraction step remains largely hand-crafted, featured by RoI pooling methods. This work proposes a general viewpoint that unifies existing region feature extraction methods and a novel method that is end-to-end learnable. The proposed method removes most heuristic choices and outperforms its RoI pooling counterparts. It moves further towards fully learnable object detection.

### (2021, Swin Transformer V1) Transformer based vision backbones (count=5)

> (2021, Swin Transformer V1) Most related to our work is the Vision Transformer (ViT) [19] and its follow-ups [60, 69, 14, 27, 63].

* [[DeiT](https://arxiv.org/abs/2012.12877)]
    [[pdf](https://arxiv.org/pdf/2012.12877.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2012.12877/)]
    * Title: Training data-efficient image transformers & distillation through attention
    * Year: 23 Dec `2020`
    * Authors: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou
    * Abstract: Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. However, these visual transformers are pre-trained with hundreds of millions of images using an expensive infrastructure, thereby limiting their adoption. In this work, we produce a competitive convolution-free transformer by training on Imagenet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop evaluation) on ImageNet with no external data. More importantly, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models.
    * Comments:
        * > (2022, Recent Advances) Touvron et al. (2020) proposed a competitive visual transformer that only trained on 1.2 million images Image-Net dataset regardless of external data.
        * > (2021, Early Convolutions) Touvron et al. show that with more regularization and stronger data augmentation ViT models achieve competitive accuracy on ImageNet-1k alone.
        * > (2021, Swin Transformer V1) While ViT requires large-scale training datasets (i.e., JFT-300M) to perform well, DeiT [60] introduces several training strategies that allow ViT to also be effective using the smaller ImageNet-1K dataset.
        * > (2021, PVT) DeiT [50] further extends ViT by using a novel distillation approach.
* [[Tokens-to-Token ViT](https://arxiv.org/abs/2101.11986)]
    [[pdf](https://arxiv.org/pdf/2101.11986.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2101.11986/)]
    * Title: Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
    * Year: 28 Jan `2021`
    * Authors: Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, Shuicheng Yan
    * Abstract: Transformers, which are popular for language modeling, have been explored for solving vision tasks recently, e.g., the Vision Transformer (ViT) for image classification. The ViT model splits each image into a sequence of tokens with fixed length and then applies multiple Transformer layers to model their global relation for classification. However, ViT achieves inferior performance to CNNs when trained from scratch on a midsize dataset like ImageNet. We find it is because: 1) the simple tokenization of input images fails to model the important local structure such as edges and lines among neighboring pixels, leading to low training sample efficiency; 2) the redundant attention backbone design of ViT leads to limited feature richness for fixed computation budgets and limited training samples. To overcome such limitations, we propose a new Tokens-To-Token Vision Transformer (T2T-ViT), which incorporates 1) a layer-wise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token (Tokens-to-Token), such that local structure represented by surrounding tokens can be modeled and tokens length can be reduced; 2) an efficient backbone with a deep-narrow structure for vision transformer motivated by CNN architecture design after empirical study. Notably, T2T-ViT reduces the parameter count and MACs of vanilla ViT by half, while achieving more than 3.0\% improvement when trained from scratch on ImageNet. It also outperforms ResNets and achieves comparable performance with MobileNets by directly training on ImageNet. For example, T2T-ViT with comparable size to ResNet50 (21.5M parameters) can achieve 83.3\% top1 accuracy in image resolution 384$\times$384 on ImageNet. (Code: this https URL)
    * Comments:
        * > (2021, Swin Transformer V1) Concurrent to our work are some that modify the ViT architecture [69, 14, 27] for better image classification.
* [[Conditional Positional Encodings for Vision Transformers](https://arxiv.org/abs/2102.10882)]
    [[pdf](https://arxiv.org/pdf/2102.10882.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2102.10882/)]
    * Title: Conditional Positional Encodings for Vision Transformers
    * Year: 22 Feb `2021`
    * Authors: Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, Xiaolin Wei, Huaxia Xia, Chunhua Shen
    * Abstract: We propose a conditional positional encoding (CPE) scheme for vision Transformers. Unlike previous fixed or learnable positional encodings, which are pre-defined and independent of input tokens, CPE is dynamically generated and conditioned on the local neighborhood of the input tokens. As a result, CPE can easily generalize to the input sequences that are longer than what the model has ever seen during training. Besides, CPE can keep the desired translation-invariance in the image classification task, resulting in improved classification accuracy. CPE can be effortlessly implemented with a simple Position Encoding Generator (PEG), and it can be seamlessly incorporated into the current Transformer framework. Built on PEG, we present Conditional Position encoding Vision Transformer (CPVT). We demonstrate that CPVT has visually similar attention maps compared to those with learned positional encodings. Benefit from the conditional positional encoding scheme, we obtain state-of-the-art results on the ImageNet classification task compared with vision Transformers to date. Our code will be made available at this https URL .
    * Comments:
        * > (2021, Swin Transformer V1) Concurrent to our work are some that modify the ViT architecture [69, 14, 27] for better image classification.
* [[Transformer in Transformer](https://arxiv.org/abs/2103.00112)]
    [[pdf](https://arxiv.org/pdf/2103.00112.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.00112/)]
    * Title: Transformer in Transformer
    * Year: 27 Feb `2021`
    * Authors: Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, Yunhe Wang
    * Abstract: Transformer is a new kind of neural architecture which encodes the input data as powerful features via the attention mechanism. Basically, the visual transformers first divide the input images into several local patches and then calculate both representations and their relationship. Since natural images are of high complexity with abundant detail and color information, the granularity of the patch dividing is not fine enough for excavating features of objects in different scales and locations. In this paper, we point out that the attention inside these local patches are also essential for building visual transformers with high performance and we explore a new architecture, namely, Transformer iN Transformer (TNT). Specifically, we regard the local patches (e.g., 16$\times$16) as "visual sentences" and present to further divide them into smaller patches (e.g., 4$\times$4) as "visual words". The attention of each word will be calculated with other words in the given visual sentence with negligible computational costs. Features of both words and sentences will be aggregated to enhance the representation ability. Experiments on several benchmarks demonstrate the effectiveness of the proposed TNT architecture, e.g., we achieve an 81.5% top-1 accuracy on the ImageNet, which is about 1.7% higher than that of the state-of-the-art visual transformer with similar computational cost. The PyTorch code is available at this https URL, and the MindSpore code is available at this https URL.
    * Comments:
        * > (2021, Swin Transformer V1) Concurrent to our work are some that modify the ViT architecture [69, 14, 27] for better image classification.
* [[PVTV1](https://arxiv.org/abs/2102.12122)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2102.12122.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2102.12122/)]
    * Title: Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
    * Year: 24 Feb `2021`
    * Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao
    * Institutions: [Nanjing University], [The University of Hong Kong], [Nanjing University of Science and Technology], [IIAI], [SenseTime Research]
    * Abstract: Although using convolutional neural networks (CNNs) as backbones achieves great successes in computer vision, this work investigates a simple backbone network useful for many dense prediction tasks without convolutions. Unlike the recently-proposed Transformer model (e.g., ViT) that is specially designed for image classification, we propose Pyramid Vision Transformer~(PVT), which overcomes the difficulties of porting Transformer to various dense prediction tasks. PVT has several merits compared to prior arts. (1) Different from ViT that typically has low-resolution outputs and high computational and memory cost, PVT can be not only trained on dense partitions of the image to achieve high output resolution, which is important for dense predictions but also using a progressive shrinking pyramid to reduce computations of large feature maps. (2) PVT inherits the advantages from both CNN and Transformer, making it a unified backbone in various vision tasks without convolutions by simply replacing CNN backbones. (3) We validate PVT by conducting extensive experiments, showing that it boosts the performance of many downstream tasks, e.g., object detection, semantic, and instance segmentation. For example, with a comparable number of parameters, RetinaNet+PVT achieves 40.4 AP on the COCO dataset, surpassing RetinNet+ResNet50 (36.3 AP) by 4.1 absolute AP. We hope PVT could serve as an alternative and useful backbone for pixel-level predictions and facilitate future researches. Code is available at this https URL.
    * Comments:
        * > (2021, Swin Transformer V1) Another concurrent work [63] explores a similar line of thinking to build multi-resolution feature maps on Transformers. Its complexity is still quadratic to image size, while ours is linear and also operates locally which has proven beneficial in modeling the high correlation in visual signals [35, 24, 40].
* [[PVTV2](https://arxiv.org/abs/2106.13797)]
    [[pdf](https://arxiv.org/pdf/2106.13797.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.13797/)]
    * Title: PVT v2: Improved Baselines with Pyramid Vision Transformer
    * Year: 25 Jun `2021`
    * Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao
    * Abstract: Transformer recently has presented encouraging progress in computer vision. In this work, we present new baselines by improving the original Pyramid Vision Transformer (PVT v1) by adding three designs, including (1) linear complexity attention layer, (2) overlapping patch embedding, and (3) convolutional feed-forward network. With these modifications, PVT v2 reduces the computational complexity of PVT v1 to linear and achieves significant improvements on fundamental vision tasks such as classification, detection, and segmentation. Notably, the proposed PVT v2 achieves comparable or better performances than recent works such as Swin Transformer. We hope this work will facilitate state-of-the-art Transformer researches in computer vision. Code is available at this https URL.

## Combine convolutions and transformers (3 + 1)

> To build robust and high-performing ViT models, hybrid approaches that combine convolutions and transformers are gaining interest. However, these hybrid models are still heavy-weight and are sensitive to data augmentation. For example, removing CutMix and DeIT-style data augmentation causes a significant drop in Image Net accuracy of Heo et al. (2021). (MobileViTv1, 2021)

* Early Convolutions Help Transformers See Better
* [[ConViT](https://arxiv.org/abs/2103.10697)]
    [[pdf](https://arxiv.org/pdf/2103.10697.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.10697/)]
    * Title: ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases
    * Year: 19 Mar `2021`
    * Authors: Stéphane d'Ascoli, Hugo Touvron, Matthew Leavitt, Ari Morcos, Giulio Biroli, Levent Sagun
    * Abstract: Convolutional architectures have proven extremely successful for vision tasks. Their hard inductive biases enable sample-efficient learning, but come at the cost of a potentially lower performance ceiling. Vision Transformers (ViTs) rely on more flexible self-attention layers, and have recently outperformed CNNs for image classification. However, they require costly pre-training on large external datasets or distillation from pre-trained convolutional networks. In this paper, we ask the following question: is it possible to combine the strengths of these two architectures while avoiding their respective limitations? To this end, we introduce gated positional self-attention (GPSA), a form of positional self-attention which can be equipped with a ``soft" convolutional inductive bias. We initialise the GPSA layers to mimic the locality of convolutional layers, then give each attention head the freedom to escape locality by adjusting a gating parameter regulating the attention paid to position versus content information. The resulting convolutional-like ViT architecture, ConViT, outperforms the DeiT on ImageNet, while offering a much improved sample efficiency. We further investigate the role of locality in learning by first quantifying how it is encouraged in vanilla self-attention layers, then analysing how it is escaped in GPSA layers. We conclude by presenting various ablations to better understand the success of the ConViT. Our code and models are released publicly at this https URL.
    * Comments:
        * > These results show that injecting some convolutional inductive bias into ViTs can be beneficial under commonly studied settings. We did not observe evidence that the hard locality constraint in early layers hampers the representational capacity of the network, as might be feared [9]. (Early Convolutions Help Transformers See Better, 2021)
        * > Evidence comes by comparison to the "hybrid ViT" presented in [13], which uses 40 convolutional layers (most of a ResNet-50) and shows no improvement over the default ViT. This perspective resonates with the findings of [9], who observe that early transformer blocks prefer to learn more local attention patterns than later blocks. (Early Convolutions Help Transformers See Better, 2021)
        * > In [9], d'Ascoli et al. modify multi-head self-attention with a convolutional bias at initialization and show that this prior improves sample efficiency and Imagenet accuracy. (Early Convolutions Help Transformers See Better, 2021)
        * > These results suggest that a (hard) convolutional bias early in the network does not compromise representational capacity, as conjectured in [9], and is beneficial within the scope of this study. (Early Convolutions Help Transformers See Better, 2021)
* [[Mobile-Former](https://arxiv.org/abs/2108.05895)]
    [[pdf](https://arxiv.org/pdf/2108.05895.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2108.05895/)]
    * Title: Mobile-Former: Bridging MobileNet and Transformer
    * Year: 12 Aug `2021`
    * Authors: Yinpeng Chen, Xiyang Dai, Dongdong Chen, Mengchen Liu, Xiaoyi Dong, Lu Yuan, Zicheng Liu
    * Abstract: We present Mobile-Former, a parallel design of MobileNet and transformer with a two-way bridge in between. This structure leverages the advantages of MobileNet at local processing and transformer at global interaction. And the bridge enables bidirectional fusion of local and global features. Different from recent works on vision transformer, the transformer in Mobile-Former contains very few tokens (e.g. 6 or fewer tokens) that are randomly initialized to learn global priors, resulting in low computational cost. Combining with the proposed light-weight cross attention to model the bridge, Mobile-Former is not only computationally efficient, but also has more representation power. It outperforms MobileNetV3 at low FLOP regime from 25M to 500M FLOPs on ImageNet classification. For instance, Mobile-Former achieves 77.9\% top-1 accuracy at 294M FLOPs, gaining 1.3\% over MobileNetV3 but saving 17\% of computations. When transferring to object detection, Mobile-Former outperforms MobileNetV3 by 8.6 AP in RetinaNet framework. Furthermore, we build an efficient end-to-end detector by replacing backbone, encoder and decoder in DETR with Mobile-Former, which outperforms DETR by 1.1 AP but saves 52\% of computational cost and 36\% of parameters.
* [[Pooling-based Vision Transformer (PiT)](https://arxiv.org/abs/2103.16302)]
    [[pdf](https://arxiv.org/pdf/2103.16302.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.16302/)]
    * Title: Rethinking Spatial Dimensions of Vision Transformers
    * Year: 30 Mar `2021`
    * Authors: Byeongho Heo, Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Junsuk Choe, Seong Joon Oh
    * Abstract: Vision Transformer (ViT) extends the application range of transformers from language processing to computer vision tasks as being an alternative architecture against the existing convolutional neural networks (CNN). Since the transformer-based architecture has been innovative for computer vision modeling, the design convention towards an effective architecture has been less studied yet. From the successful design principles of CNN, we investigate the role of spatial dimension conversion and its effectiveness on transformer-based architecture. We particularly attend to the dimension reduction principle of CNNs; as the depth increases, a conventional CNN increases channel dimension and decreases spatial dimensions. We empirically show that such a spatial dimension reduction is beneficial to a transformer architecture as well, and propose a novel Pooling-based Vision Transformer (PiT) upon the original ViT model. We show that PiT achieves the improved model capability and generalization performance against ViT. Throughout the extensive experiments, we further show PiT outperforms the baseline on several tasks such as image classification, object detection, and robustness evaluation. Source codes and ImageNet models are available at this https URL

## Improving Transformer-Based Models (MobileViTv2)

* Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
* MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
* Delight: Deep and light-weight transformer
* CvT: Introducing Convolutions to Vision Transformers
* Rethinking Spatial Dimensions of Vision Transformers
* [TokenLearner](https://openreview.net/forum?id=z-l1kpDXs88)
    * Title: TokenLearner: Adaptive Space-Time Tokenization for Videos
    * Year: 21 May `2021`
    * Authors: Michael S Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, Anelia Angelova
    * Abstract: In this paper, we introduce a novel visual representation learning which relies on a handful of adaptively learned tokens, and which is applicable to both image and video understanding tasks. Instead of relying on hand-designed splitting strategies to obtain visual tokens and processing a large number of densely sampled patches for attention, our approach learns to mine important tokens in visual data. This results in efficiently and effectively finding a few important visual tokens and enables modeling of pairwise attention between such tokens, over a longer temporal horizon for videos, or the spatial content in image frames. Our experiments demonstrate strong performance on several challenging benchmarks for video recognition tasks. Importantly, due to our tokens being adaptive, we accomplish competitive results at significantly reduced computational cost. We establish new state-of-the-arts on multiple video datasets, including Kinetics-400, Kinetics-600, Charades, and AViD.
* Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions

## Continuous convolution and variants (4)

* Relation Networks for Object Detection
* [[A Closer Look at Local Aggregation Operators in Point Cloud Analysis](https://arxiv.org/abs/2007.01294)]
    [[pdf](https://arxiv.org/pdf/2007.01294.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2007.01294/)]
    * Title: A Closer Look at Local Aggregation Operators in Point Cloud Analysis
    * Year: 02 Jul `2020`
    * Authors: Ze Liu, Han Hu, Yue Cao, Zheng Zhang, Xin Tong
    * Abstract: Recent advances of network architecture for point cloud processing are mainly driven by new designs of local aggregation operators. However, the impact of these operators to network performance is not carefully investigated due to different overall network architecture and implementation details in each solution. Meanwhile, most of operators are only applied in shallow architectures. In this paper, we revisit the representative local aggregation operators and study their performance using the same deep residual architecture. Our investigation reveals that despite the different designs of these operators, all of these operators make surprisingly similar contributions to the network performance under the same network input and feature numbers and result in the state-of-the-art accuracy on standard benchmarks. This finding stimulate us to rethink the necessity of sophisticated design of local aggregation operator for point cloud processing. To this end, we propose a simple local aggregation operator without learnable weights, named Position Pooling (PosPool), which performs similarly or slightly better than existing sophisticated operators. In particular, a simple deep residual network with PosPool layers achieves outstanding performance on all benchmarks, which outperforms the previous state-of-the methods on the challenging PartNet datasets by a large margin (7.4 mIoU). The code is publicly available at this https URL
* [[SchNet](https://arxiv.org/abs/1706.08566)]
    [[pdf](https://arxiv.org/pdf/1706.08566.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1706.08566/)]
    * Title: SchNet: A continuous-filter convolutional neural network for modeling quantum interactions
    * Year: 26 Jun `2017`
    * Authors: Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko, Klaus-Robert Müller
    * Abstract: Deep learning has the potential to revolutionize quantum chemistry as it is ideally suited to learn representations for structured data and speed up the exploration of chemical space. While convolutional neural networks have proven to be the first choice for images, audio and video data, the atoms in molecules are not restricted to a grid. Instead, their precise locations contain essential physical information, that would get lost if discretized. Thus, we propose to use continuous-filter convolutional layers to be able to model local correlations without requiring the data to lie on a grid. We apply those layers in SchNet: a novel deep learning architecture modeling quantum interactions in molecules. We obtain a joint model for the total energy and interatomic forces that follows fundamental quantum-chemical principles. This includes rotationally invariant energy predictions and a smooth, differentiable potential energy surface. Our architecture achieves state-of-the-art performance for benchmarks of equilibrium molecules and molecular dynamics trajectories. Finally, we introduce a more challenging benchmark with chemical and structural variations that suggests the path for further work.
* [[Deep Parametric Continuous Convolutional Neural Networks](https://arxiv.org/abs/2101.06742)]
    [[pdf](https://arxiv.org/pdf/2101.06742.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2101.06742/)]
    * Title: Deep Parametric Continuous Convolutional Neural Networks
    * Year: 17 Jan `2021`
    * Authors: Shenlong Wang, Simon Suo, Wei-Chiu Ma, Andrei Pokrovsky, Raquel Urtasun
    * Abstract: Standard convolutional neural networks assume a grid structured input is available and exploit discrete convolutions as their fundamental building blocks. This limits their applicability to many real-world applications. In this paper we propose Parametric Continuous Convolution, a new learnable operator that operates over non-grid structured data. The key idea is to exploit parameterized kernel functions that span the full continuous vector space. This generalization allows us to learn over arbitrary data structures as long as their support relationship is computable. Our experiments show significant improvement over the state-of-the-art in point cloud segmentation of indoor and outdoor scenes, and lidar motion estimation of driving scenes.

## Sliding Window Approaches

* Local Relation Networks for Image Recognition
* Stand-Alone Self-Attention in Vision Models

## Dictionary lookup problem (2021, PVT) (count=5)

> (2021, PVT) Some works [4, 64, 55, 43, 17] model the vision task as a dictionary lookup problem with learnable queries, and use the Transformer decoder as a task-specific head on the top of the CNN backbone, such as VGG [41] and ResNet [15].

* DETR
* Deformable DETR
* [[Segmenting Transparent Object in the Wild with Transformer](https://arxiv.org/abs/2101.08461)]
    [[pdf](https://arxiv.org/pdf/2101.08461.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2101.08461/)]
    * Title: Segmenting Transparent Object in the Wild with Transformer
    * Year: 21 Jan `2021`
    * Authors: Enze Xie, Wenjia Wang, Wenhai Wang, Peize Sun, Hang Xu, Ding Liang, Ping Luo
    * Abstract: This work presents a new fine-grained transparent object segmentation dataset, termed Trans10K-v2, extending Trans10K-v1, the first large-scale transparent object segmentation dataset. Unlike Trans10K-v1 that only has two limited categories, our new dataset has several appealing benefits. (1) It has 11 fine-grained categories of transparent objects, commonly occurring in the human domestic environment, making it more practical for real-world application. (2) Trans10K-v2 brings more challenges for the current advanced segmentation methods than its former version. Furthermore, a novel transformer-based segmentation pipeline termed Trans2Seg is proposed. Firstly, the transformer encoder of Trans2Seg provides the global receptive field in contrast to CNN's local receptive field, which shows excellent advantages over pure CNN architectures. Secondly, by formulating semantic segmentation as a problem of dictionary look-up, we design a set of learnable prototypes as the query of Trans2Seg's transformer decoder, where each prototype learns the statistics of one category in the whole dataset. We benchmark more than 20 recent semantic segmentation methods, demonstrating that Trans2Seg significantly outperforms all the CNN-based methods, showing the proposed algorithm's potential ability to solve transparent object segmentation.
* [[TransTrack](https://arxiv.org/abs/2012.15460)]
    [[pdf](https://arxiv.org/pdf/2012.15460.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2012.15460/)]
    * Title: TransTrack: Multiple Object Tracking with Transformer
    * Year: 31 Dec `2020`
    * Authors: Peize Sun, Jinkun Cao, Yi Jiang, Rufeng Zhang, Enze Xie, Zehuan Yuan, Changhu Wang, Ping Luo
    * Abstract: In this work, we propose TransTrack, a simple but efficient scheme to solve the multiple object tracking problems. TransTrack leverages the transformer architecture, which is an attention-based query-key mechanism. It applies object features from the previous frame as a query of the current frame and introduces a set of learned object queries to enable detecting new-coming objects. It builds up a novel joint-detection-and-tracking paradigm by accomplishing object detection and object association in a single shot, simplifying complicated multi-step settings in tracking-by-detection methods. On MOT17 and MOT20 benchmark, TransTrack achieves 74.5\% and 64.5\% MOTA, respectively, competitive to the state-of-the-art methods. We expect TransTrack to provide a novel perspective for multiple object tracking. The code is available at: \url{this https URL}.
* [[UniT](https://arxiv.org/abs/2102.10772)]
    [[pdf](https://arxiv.org/pdf/2102.10772.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2102.10772/)]
    * Title: UniT: Multimodal Multitask Learning with a Unified Transformer
    * Year: 22 Feb `2021`
    * Authors: Ronghang Hu, Amanpreet Singh
    * Abstract: We propose UniT, a Unified Transformer model to simultaneously learn the most prominent tasks across different domains, ranging from object detection to natural language understanding and multimodal reasoning. Based on the transformer encoder-decoder architecture, our UniT model encodes each input modality with an encoder and makes predictions on each task with a shared decoder over the encoded input representations, followed by task-specific output heads. The entire model is jointly trained end-to-end with losses from each task. Compared to previous efforts on multi-task learning with transformers, we share the same model parameters across all tasks instead of separately fine-tuning task-specific models and handle a much higher variety of tasks across different domains. In our experiments, we learn 7 tasks jointly over 8 datasets, achieving strong performance on each task with significantly fewer parameters. Our code is available in MMF at this https URL.

* [[Visual Saliency Transformer](https://arxiv.org/abs/2104.12099)]
    [[pdf](https://arxiv.org/pdf/2104.12099.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2104.12099/)]
    * Title: Visual Saliency Transformer
    * Year: 25 Apr `2021`
    * Authors: Nian Liu, Ni Zhang, Kaiyuan Wan, Ling Shao, Junwei Han
    * Abstract: Existing state-of-the-art saliency detection methods heavily rely on CNN-based architectures. Alternatively, we rethink this task from a convolution-free sequence-to-sequence perspective and predict saliency by modeling long-range dependencies, which can not be achieved by convolution. Specifically, we develop a novel unified model based on a pure transformer, namely, Visual Saliency Transformer (VST), for both RGB and RGB-D salient object detection (SOD). It takes image patches as inputs and leverages the transformer to propagate global contexts among image patches. Unlike conventional architectures used in Vision Transformer (ViT), we leverage multi-level token fusion and propose a new token upsampling method under the transformer framework to get high-resolution detection results. We also develop a token-based multi-task decoder to simultaneously perform saliency and boundary detection by introducing task-related tokens and a novel patch-task-attention mechanism. Experimental results show that our model outperforms existing methods on both RGB and RGB-D SOD benchmark datasets. Most importantly, our whole framework not only provides a new perspective for the SOD field but also shows a new paradigm for transformer-based dense prediction models. Code is available at this https URL.

## Substandard Optimizability is Due to the Lack of Spatial Inductive Biases in ViTs.

> However, unlike CNNs, ViTs show substandard optimizability and are difficult to train. Subsequenct works shows that this substandard optimizability is due to the lack of spatial inductive biases in ViTs. Incorporating such biases using convolutions in ViTs improves their stability and performance. (MobileViT, 2021)
* LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference
* CoAtNet: Marrying Convolution and Attention for All Data Sizes
* Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
* Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
* Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
* Mobile-Former: Bridging MobileNet and Transformer

> Different designs have been explored to reap the benefits of convolutions and transformers.
> * ViT-C of Xiao et al. (2021) adds an early convlution stem to ViT.
> * CvT (Wu et al., 2021) modifies the multi-head attention in transformers and uses depth-wise separable convolutions instead of linear projections.
> * BoTNet (Srinivas et al., 2021) replaces the standard 3x3 convolution in the bottleneck unit of ResNetwith multi-head attention.
> * ConViT (d'Ascoli et al., 2021) incorporates soft convolutional inductive biases using a gated positional self-attention.
> * PiT (Heo et al, 2021) extends ViT with depth-wise convolution-based pooling layer.
> Though these models can achieve competitive performance to CNNs with extensive augmentation, the majority of these models are heavy-weight.
> Also, when these models are scaled down to build light-weight ViT models, their performance is significantly worse than light-weight CNNs. (MobileViT, 2021)

## Transformer Architecture Applied to Object Detection and Instance Segmentation (4)

> More recently, the encoder-decoder design in Transformer has been applied for the object detection and instance segmentation tasks. (Swin Transformer V1, 2021)

* DETR (see detection.md)
* [[RelationNet++](https://arxiv.org/abs/2010.15831)]
    [[pdf](https://arxiv.org/pdf/2010.15831.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2010.15831/)]
    * Title: RelationNet++: Bridging Visual Representations for Object Detection via Transformer Decoder
    * Year: 29 Oct `2020`
    * Authors: Cheng Chi, Fangyun Wei, Han Hu
    * Abstract: Existing object detection frameworks are usually built on a single format of object/part representation, i.e., anchor/proposal rectangle boxes in RetinaNet and Faster R-CNN, center points in FCOS and RepPoints, and corner points in CornerNet. While these different representations usually drive the frameworks to perform well in different aspects, e.g., better classification or finer localization, it is in general difficult to combine these representations in a single framework to make good use of each strength, due to the heterogeneous or non-grid feature extraction by different representations. This paper presents an attention-based decoder module similar as that in Transformer~\cite{vaswani2017attention} to bridge other representations into a typical object detector built on a single representation format, in an end-to-end fashion. The other representations act as a set of \emph{key} instances to strengthen the main \emph{query} representation features in the vanilla detectors. Novel techniques are proposed towards efficient computation of the decoder module, including a \emph{key sampling} approach and a \emph{shared location embedding} approach. The proposed module is named \emph{bridging visual representations} (BVR). It can perform in-place and we demonstrate its broad effectiveness in bridging other representations into prevalent object detection frameworks, including RetinaNet, Faster R-CNN, FCOS and ATSS, where about $1.5\sim3.0$ AP improvements are achieved. In particular, we improve a state-of-the-art framework with a strong backbone by about $2.0$ AP, reaching $52.7$ AP on COCO test-dev. The resulting network is named RelationNet++. The code will be available at this https URL.
* Deformable DETR (see detection.md)
* Sparse R-CNN (see detection_2D.md)

## MobileViT

> MobileViT is a hybrid netowrk that combines the strengths of CNNs and ViTs. MobileViT views transormers as cconvoluions which allows it to leverage the merits of both convolutions (e.g., inductive biases) and transformers (e.g., long-range dependencies) to build a light-weight network for mobile devices. Though MobileViT networks have significtantly fewer parameters and deliver better performance as compared to light-weight CNNs (e.g., MobileNets), they have high latency. The main efficiency botttleneck in MobileViT is the multi-headed self-attention. (MobileViTv2, 2022)

* [[MobileViTv1](https://arxiv.org/abs/2110.02178)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2110.02178.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2110.02178/)]
    * Title: MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
    * Year: 05 Oct `2021`
    * Authors: Sachin Mehta, Mohammad Rastegari
    * Abstract: Light-weight convolutional neural networks (CNNs) are the de-facto for mobile vision tasks. Their spatial inductive biases allow them to learn representations with fewer parameters across different vision tasks. However, these networks are spatially local. To learn global representations, self-attention-based vision trans-formers (ViTs) have been adopted. Unlike CNNs, ViTs are heavy-weight. In this paper, we ask the following question: is it possible to combine the strengths of CNNs and ViTs to build a light-weight and low latency network for mobile vision tasks? Towards this end, we introduce MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. Our results show that MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters. Our source code is open-source and available at: this https URL
* [[MobileViTv2](https://arxiv.org/abs/2206.02680)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2206.02680.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2206.02680/)]
    * Title: Separable Self-attention for Mobile Vision Transformers
    * Year: 06 Jun `2022`
    * Authors: Sachin Mehta, Mohammad Rastegari
    * Abstract: Mobile vision transformers (MobileViT) can achieve state-of-the-art performance across several mobile vision tasks, including classification and detection. Though these models have fewer parameters, they have high latency as compared to convolutional neural network-based models. The main efficiency bottleneck in MobileViT is the multi-headed self-attention (MHA) in transformers, which requires $O(k^2)$ time complexity with respect to the number of tokens (or patches) $k$. Moreover, MHA requires costly operations (e.g., batch-wise matrix multiplication) for computing self-attention, impacting latency on resource-constrained devices. This paper introduces a separable self-attention method with linear complexity, i.e. $O(k)$. A simple yet effective characteristic of the proposed method is that it uses element-wise operations for computing self-attention, making it a good choice for resource-constrained devices. The improved model, MobileViTv2, is state-of-the-art on several mobile vision tasks, including ImageNet object classification and MS-COCO object detection. With about three million parameters, MobileViTv2 achieves a top-1 accuracy of 75.6% on the ImageNet dataset, outperforming MobileViT by about 1% while running $3.2\times$ faster on a mobile device. Our source code is available at: \url{this https URL}


## Swin Transformer

> A key design element of Swin Transformer is its shift of the window partition between consecutive self-attention layers. The shifted windows bridge the windows of the preceding layer, providing connections among them that significantly enhance modeling power. This strategy is also efficient in regards to real-world latency: all query patches within a window share the same key set, which facilitates memory access in hardware. In contrast, earlier sliding window based self-attention approaches suffer from low latency on general hardware due to different key sets for different query pixels. Our experiments show that the proposed *shifted window* approach has much lower latency than the *sliding window* method, yet is similar in modeling power. The shifted window approach also proves beneficial for all-MLP architectures. (Swin Transformer V1, 2021)

* [[Swin Transformer V1](https://arxiv.org/abs/2103.14030)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2103.14030.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.14030/)]
    * Title: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    * Year: 25 Mar `2021`
    * Authors: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
    * Institutions: [Microsoft Research Asia]
    * Abstract: This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \textbf{S}hifted \textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at~\url{this https URL}.
    * Comments:
        * > (2021, Swin Transformer V2) Swin Transformer is a general-purpose computer vision backbone that has achieved strong performance in various granular recognition tasks such as region-level object detection, pixel-level semantic segmentation, and image-level image classification. The main idea of Swin Transformer is to introduce several important visual priors into the vanilla Transformer encoder, including hierarchy, locality, and translation invariance, which combines the strength of both: the basic Transformer unit has strong modeling capabilities, and the visual priors make it friendly to a variety of visual tasks.
* [[Swin Transformer V2](https://arxiv.org/abs/2111.09883)] <!-- printed -->
    [[pdf](https://arxiv.org/pdf/2111.09883.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2111.09883/)]
    * Title: Swin Transformer V2: Scaling Up Capacity and Resolution
    * Year: 18 Nov `2021`
    * Authors: Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo
    * Institutions: [Microsoft Research Asia]
    * Abstract: Large-scale NLP models have been shown to significantly improve the performance on language tasks with no signs of saturation. They also demonstrate amazing few-shot capabilities like that of human beings. This paper aims to explore large-scale models in computer vision. We tackle three major issues in training and application of large vision models, including training instability, resolution gaps between pre-training and fine-tuning, and hunger on labelled data. Three main techniques are proposed: 1) a residual-post-norm method combined with cosine attention to improve training stability; 2) A log-spaced continuous position bias method to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs; 3) A self-supervised pre-training method, SimMIM, to reduce the needs of vast labeled images. Through these techniques, this paper successfully trained a 3 billion-parameter Swin Transformer V2 model, which is the largest dense vision model to date, and makes it capable of training with images of up to 1,536$\times$1,536 resolution. It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification. Also note our training is much more efficient than that in Google's billion-level visual models, which consumes 40 times less labelled data and 40 times less training time. Code is available at \url{this https URL}.

## Others

* [[Dynamic Head](https://arxiv.org/abs/2106.08322)]
    [[pdf](https://arxiv.org/pdf/2106.08322.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2106.08322/)]
    * Title: Dynamic Head: Unifying Object Detection Heads with Attentions
    * Year: 15 Jun `2021`
    * Authors: Xiyang Dai, Yinpeng Chen, Bin Xiao, Dongdong Chen, Mengchen Liu, Lu Yuan, Lei Zhang
    * Abstract: The complex nature of combining localization and classification in object detection has resulted in the flourished development of methods. Previous works tried to improve the performance in various object detection heads but failed to present a unified view. In this paper, we present a novel dynamic head framework to unify object detection heads with attentions. By coherently combining multiple self-attention mechanisms between feature levels for scale-awareness, among spatial locations for spatial-awareness, and within output channels for task-awareness, the proposed approach significantly improves the representation ability of object detection heads without any computational overhead. Further experiments demonstrate that the effectiveness and efficiency of the proposed dynamic head on the COCO benchmark. With a standard ResNeXt-101-DCN backbone, we largely improve the performance over popular object detectors and achieve a new state-of-the-art at 54.0 AP. Furthermore, with latest transformer backbone and extra data, we can push current best COCO result to a new record at 60.6 AP. The code will be released at this https URL.
* [[Fastformer](https://arxiv.org/abs/2108.09084)]
    [[pdf](https://arxiv.org/pdf/2108.09084.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2108.09084/)]
    * Title: Fastformer: Additive Attention Can Be All You Need
    * Year: 20 Aug `2021`
    * Authors: Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang, Xing Xie
    * Abstract: Transformer is a powerful model for text understanding. However, it is inefficient due to its quadratic complexity to input sequence length. Although there are many methods on Transformer acceleration, they are still either inefficient on long sequences or not effective enough. In this paper, we propose Fastformer, which is an efficient Transformer model based on additive attention. In Fastformer, instead of modeling the pair-wise interactions between tokens, we first use additive attention mechanism to model global contexts, and then further transform each token representation based on its interaction with global context representations. In this way, Fastformer can achieve effective context modeling with linear complexity. Extensive experiments on five datasets show that Fastformer is much more efficient than many existing Transformer models and can meanwhile achieve comparable or even better long text modeling performance.
* [[Mask DINO](https://arxiv.org/abs/2206.02777)]
    [[pdf](https://arxiv.org/pdf/2206.02777.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2206.02777/)]
    * Title: Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation
    * Year: 06 Jun `2022`
    * Authors: Feng Li, Hao Zhang, Huaizhe xu, Shilong Liu, Lei Zhang, Lionel M. Ni, Heung-Yeung Shum
    * Abstract: In this paper we present Mask DINO, a unified object detection and segmentation framework. Mask DINO extends DINO (DETR with Improved Denoising Anchor Boxes) by adding a mask prediction branch which supports all image segmentation tasks (instance, panoptic, and semantic). It makes use of the query embeddings from DINO to dot-product a high-resolution pixel embedding map to predict a set of binary masks. Some key components in DINO are extended for segmentation through a shared architecture and training process. Mask DINO is simple, efficient, scalable, and benefits from joint large-scale detection and segmentation datasets. Our experiments show that Mask DINO significantly outperforms all existing specialized segmentation methods, both on a ResNet-50 backbone and a pre-trained model with SwinL backbone. Notably, Mask DINO establishes the best results to date on instance segmentation (54.5 AP on COCO), panoptic segmentation (59.4 PQ on COCO), and semantic segmentation (60.8 mIoU on ADE20K). Code will be avaliable at \url{this https URL}.
* [[Aggregated Pyramid Vision Transformer](https://arxiv.org/abs/2203.00960)]
    [[pdf](https://arxiv.org/pdf/2203.00960.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2203.00960/)]
    * Title: Aggregated Pyramid Vision Transformer: Split-transform-merge Strategy for Image Recognition without Convolutions
    * Year: 02 Mar `2022`
    * Authors: Rui-Yang Ju, Ting-Yu Lin, Jen-Shiun Chiang, Jia-Hao Jian, Yu-Shian Lin, Liu-Rui-Yi Huang
    * Abstract: With the achievements of Transformer in the field of natural language processing, the encoder-decoder and the attention mechanism in Transformer have been applied to computer vision. Recently, in multiple tasks of computer vision (image classification, object detection, semantic segmentation, etc.), state-of-the-art convolutional neural networks have introduced some concepts of Transformer. This proves that Transformer has a good prospect in the field of image recognition. After Vision Transformer was proposed, more and more works began to use self-attention to completely replace the convolutional layer. This work is based on Vision Transformer, combined with the pyramid architecture, using Split-transform-merge to propose the group encoder and name the network architecture Aggregated Pyramid Vision Transformer (APVT). We perform image classification tasks on the CIFAR-10 dataset and object detection tasks on the COCO 2017 dataset. Compared with other network architectures that use Transformer as the backbone, APVT has excellent results while reducing the computational cost. We hope this improved strategy can provide a reference for future Transformer research in computer vision.
