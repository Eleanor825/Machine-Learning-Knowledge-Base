# [Papers][Vision] GAN

count: 6

* [GAN](https://arxiv.org/abs/1406.2661)
    * Title: Generative Adversarial Networks
    * Year: 10 Jun `2014`
    * Author: Ian J. Goodfellow
    * Abstract: We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.
* [StyleGAN](https://arxiv.org/abs/1812.04948)
    * Title: A Style-Based Generator Architecture for Generative Adversarial Networks
    * Year: 12 Dec `2018`
    * Author: Tero Karras
    * Abstract: We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.
* [StyleGANv2](https://arxiv.org/abs/1912.04958)
    * Title: Analyzing and Improving the Image Quality of StyleGAN
    * Year: 03 Dec `2019`
    * Author: Tero Karras
    * Abstract: The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven unconditional generative image modeling. We expose and analyze several of its characteristic artifacts, and propose changes in both model architecture and training methods to address them. In particular, we redesign the generator normalization, revisit progressive growing, and regularize the generator to encourage good conditioning in the mapping from latent codes to images. In addition to improving image quality, this path length regularizer yields the additional benefit that the generator becomes significantly easier to invert. This makes it possible to reliably attribute a generated image to a particular network. We furthermore visualize how well the generator utilizes its output resolution, and identify a capacity problem, motivating us to train larger models for additional quality improvements. Overall, our improved model redefines the state of the art in unconditional image modeling, both in terms of existing distribution quality metrics as well as perceived image quality.
* [DualStyleGAN](https://arxiv.org/abs/2203.13248)
    * Title: Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer
    * Year: 24 Mar `2022`
    * Author: Shuai Yang
    * Abstract: Recent studies on StyleGAN show high performance on artistic portrait generation by transfer learning with limited data. In this paper, we explore more challenging exemplar-based high-resolution portrait style transfer by introducing a novel DualStyleGAN with flexible control of dual styles of the original face domain and the extended artistic portrait domain. Different from StyleGAN, DualStyleGAN provides a natural way of style transfer by characterizing the content and style of a portrait with an intrinsic style path and a new extrinsic style path, respectively. The delicately designed extrinsic style path enables our model to modulate both the color and complex structural styles hierarchically to precisely pastiche the style example. Furthermore, a novel progressive fine-tuning scheme is introduced to smoothly transform the generative space of the model to the target domain, even with the above modifications on the network architecture. Experiments demonstrate the superiority of DualStyleGAN over state-of-the-art methods in high-quality portrait style transfer and flexible style control.
* [Generative Multiplane Images: Making a 2D GAN 3D-Aware](https://arxiv.org/abs/2207.10642)
    * Title: Generative Multiplane Images: Making a 2D GAN 3D-Aware
    * Year: 21 Jul `2022`
    * Author: Xiaoming Zhao
    * Abstract: What is really needed to make an existing 2D GAN 3D-aware? To answer this question, we modify a classical GAN, i.e., StyleGANv2, as little as possible. We find that only two modifications are absolutely necessary: 1) a multiplane image style generator branch which produces a set of alpha maps conditioned on their depth; 2) a pose-conditioned discriminator. We refer to the generated output as a 'generative multiplane image' (GMPI) and emphasize that its renderings are not only high-quality but also guaranteed to be view-consistent, which makes GMPIs different from many prior works. Importantly, the number of alpha maps can be dynamically adjusted and can differ between training and inference, alleviating memory concerns and enabling fast training of GMPIs in less than half a day at a resolution of $1024^{2}$. Our findings are consistent across three challenging and common high-resolution datasets, including FFHQ, AFHQv2, and MetFaces.
* [A Survey of Explainable Graph Neural Networks: Taxonomy and Evaluation Metrics](https://arxiv.org/abs/2207.12599)
    * Title: A Survey of Explainable Graph Neural Networks: Taxonomy and Evaluation Metrics
    * Year: 26 Jul `2022`
    * Author: Yiqiao Li
    * Abstract: Graph neural networks (GNNs) have demonstrated a significant boost in prediction performance on graph data. At the same time, the predictions made by these models are often hard to interpret. In that regard, many efforts have been made to explain the prediction mechanisms of these models from perspectives such as GNNExplainer, XGNN and PGExplainer. Although such works present systematic frameworks to interpret GNNs, a holistic review for explainable GNNs is unavailable. In this survey, we present a comprehensive review of explainability techniques developed for GNNs. We focus on explainable graph neural networks and categorize them based on the use of explainable methods. We further provide the common performance metrics for GNNs explanations and point out several future research directions.

## Progressive Learning (EfficientNetV2, 2021)

* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
    * Title: Progressive Growing of GANs for Improved Quality, Stability, and Variation
    * Year: 27 Oct `2017`
    * Authors: Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
    * Abstract: We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024^2. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CelebA dataset.

## Adversarial Learning (EfficientNetV2, 2021)

* [PDA](https://arxiv.org/abs/1909.04839)
    * Title: PDA: Progressive Data Augmentation for General Robustness of Deep Neural Networks
    * Year: 11 Sep `2019`
    * Authors: Hang Yu, Aishan Liu, Xianglong Liu, Gengchao Li, Ping Luo, Ran Cheng, Jichen Yang, Chongzhi Zhang
    * Abstract: Adversarial images are designed to mislead deep neural networks (DNNs), attracting great attention in recent years. Although several defense strategies achieved encouraging robustness against adversarial samples, most of them fail to improve the robustness on common corruptions such as noise, blur, and weather/digital effects (e.g. frost, pixelate). To address this problem, we propose a simple yet effective method, named Progressive Data Augmentation (PDA), which enables general robustness of DNNs by progressively injecting diverse adversarial noises during training. In other words, DNNs trained with PDA are able to obtain more robustness against both adversarial attacks as well as common corruptions than the recent state-of-the-art methods. We also find that PDA is more efficient than prior arts and able to prevent accuracy drop on clean samples without being attacked. Furthermore, we theoretically show that PDA can control the perturbation bound and guarantee better generalization ability than existing work. Extensive experiments on many benchmarks such as CIFAR-10, SVHN, and ImageNet demonstrate that PDA significantly outperforms its counterparts in various experimental setups.
