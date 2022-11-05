# [Papers][Vision] GAN <!-- omit in toc -->

count=48

## Table of Contents <!-- omit in toc -->

- [Basics](#basics)
- [StyleGAN Family](#stylegan-family)
- [Variational Autoencoder Related](#variational-autoencoder-related)
- [Wasserstein GAN Related](#wasserstein-gan-related)
- [Auto-Regressive Generative Model](#auto-regressive-generative-model)
- [Adversarial Learning (EfficientNetV2, 2021)](#adversarial-learning-efficientnetv2-2021)
- [Diffusion Models](#diffusion-models)
- [Text-to-Image Models](#text-to-image-models)
- [Unclassified](#unclassified)

----------------------------------------------------------------------------------------------------

## Basics

* [[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)]
    [[pdf](https://arxiv.org/pdf/1406.2661.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1406.2661/)]
    * Title: Generative Adversarial Networks
    * Year: 10 Jun `2014`
    * Authors: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
    * Abstract: We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.
* [[Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)]
    [[pdf](https://arxiv.org/pdf/1903.07291.pdf)]
    * Title: Semantic Image Synthesis with Spatially-Adaptive Normalization
    * Year: 18 Mar `2019`
    * Authors: Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu
    * Abstract: We propose spatially-adaptive normalization, a simple but effective layer for synthesizing photorealistic images given an input semantic layout. Previous methods directly feed the semantic layout as input to the deep network, which is then processed through stacks of convolution, normalization, and nonlinearity layers. We show that this is suboptimal as the normalization layers tend to ``wash away'' semantic information. To address the issue, we propose using the input layout for modulating the activations in normalization layers through a spatially-adaptive, learned transformation. Experiments on several challenging datasets demonstrate the advantage of the proposed method over existing approaches, regarding both visual fidelity and alignment with input layouts. Finally, our model allows user control over both semantic and style. Code is available at this https URL .
* [[CycleGAN](https://arxiv.org/abs/1703.10593)]
    [[pdf](https://arxiv.org/pdf/1703.10593.pdf)]
    * Title: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
    * Year: 30 Mar `2017`
    * Authors: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
    * Abstract: Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.
* [[Pix2Pix](https://arxiv.org/abs/1611.07004)]
    [[pdf](https://arxiv.org/pdf/1611.07004.pdf)]
    * Title: Image-to-Image Translation with Conditional Adversarial Networks
    * Year: 21 Nov `2016`
    * Authors: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
    * Abstract: We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.
* [[Deep Convolutional GAN](https://arxiv.org/abs/1511.06434)]
    [[pdf](https://arxiv.org/pdf/1511.06434.pdf)]
    * Title: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    * Year: 19 Nov `2015`
    * Authors: Alec Radford, Luke Metz, Soumith Chintala
    * Abstract: In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.
* [[StackGAN](https://arxiv.org/abs/1612.03242)]
    [[pdf](https://arxiv.org/pdf/1612.03242.pdf)]
    * Title: StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks
    * Year: 10 Dec `2016`
    * Authors: Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas
    * Abstract: Synthesizing high-quality images from text descriptions is a challenging problem in computer vision and has many practical applications. Samples generated by existing text-to-image approaches can roughly reflect the meaning of the given descriptions, but they fail to contain necessary details and vivid object parts. In this paper, we propose Stacked Generative Adversarial Networks (StackGAN) to generate 256x256 photo-realistic images conditioned on text descriptions. We decompose the hard problem into more manageable sub-problems through a sketch-refinement process. The Stage-I GAN sketches the primitive shape and colors of the object based on the given text description, yielding Stage-I low-resolution images. The Stage-II GAN takes Stage-I results and text descriptions as inputs, and generates high-resolution images with photo-realistic details. It is able to rectify defects in Stage-I results and add compelling details with the refinement process. To improve the diversity of the synthesized images and stabilize the training of the conditional-GAN, we introduce a novel Conditioning Augmentation technique that encourages smoothness in the latent conditioning manifold. Extensive experiments and comparisons with state-of-the-arts on benchmark datasets demonstrate that the proposed method achieves significant improvements on generating photo-realistic images conditioned on text descriptions.
* [[Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)]
    [[pdf](https://arxiv.org/pdf/1809.11096.pdf)]
    * Title: Large Scale GAN Training for High Fidelity Natural Image Synthesis
    * Year: 28 Sep `2018`
    * Authors: Andrew Brock, Jeff Donahue, Karen Simonyan
    * Abstract: Despite recent progress in generative image modeling, successfully generating high-resolution, diverse samples from complex datasets such as ImageNet remains an elusive goal. To this end, we train Generative Adversarial Networks at the largest scale yet attempted, and study the instabilities specific to such scale. We find that applying orthogonal regularization to the generator renders it amenable to a simple "truncation trick," allowing fine control over the trade-off between sample fidelity and variety by reducing the variance of the Generator's input. Our modifications lead to models which set the new state of the art in class-conditional image synthesis. When trained on ImageNet at 128x128 resolution, our models (BigGANs) achieve an Inception Score (IS) of 166.5 and Frechet Inception Distance (FID) of 7.4, improving over the previous best IS of 52.52 and FID of 18.6.
* [[Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)]
    [[pdf](https://arxiv.org/pdf/1606.03498.pdf)]
    * Title: Improved Techniques for Training GANs
    * Year: 10 Jun `2016`
    * Authors: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
    * Abstract: We present a variety of new architectural features and training procedures that we apply to the generative adversarial networks (GANs) framework. We focus on two applications of GANs: semi-supervised learning, and the generation of images that humans find visually realistic. Unlike most work on generative models, our primary goal is not to train a model that assigns high likelihood to test data, nor do we require the model to be able to learn well without using any labels. Using our new techniques, we achieve state-of-the-art results in semi-supervised classification on MNIST, CIFAR-10 and SVHN. The generated images are of high quality as confirmed by a visual Turing test: our model generates MNIST samples that humans cannot distinguish from real data, and CIFAR-10 samples that yield a human error rate of 21.3%. We also present ImageNet samples with unprecedented resolution and show that our methods enable the model to learn recognizable features of ImageNet classes.
* [[Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)]
    [[pdf](https://arxiv.org/pdf/1802.05957.pdf)]
    * Title: Spectral Normalization for Generative Adversarial Networks
    * Year: 16 Feb `2018`
    * Authors: Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    * Abstract: One of the challenges in the study of generative adversarial networks is the instability of its training. In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator. Our new normalization technique is computationally light and easy to incorporate into existing implementations. We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques.
* [[Self-Attention GAN](https://arxiv.org/abs/1805.08318)]
    [[pdf](https://arxiv.org/pdf/1805.08318.pdf)]
    * Title: Self-Attention Generative Adversarial Networks
    * Year: 21 May `2018`
    * Authors: Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena
    * Abstract: In this paper, we propose the Self-Attention Generative Adversarial Network (SAGAN) which allows attention-driven, long-range dependency modeling for image generation tasks. Traditional convolutional GANs generate high-resolution details as a function of only spatially local points in lower-resolution feature maps. In SAGAN, details can be generated using cues from all feature locations. Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other. Furthermore, recent work has shown that generator conditioning affects GAN performance. Leveraging this insight, we apply spectral normalization to the GAN generator and find that this improves training dynamics. The proposed SAGAN achieves the state-of-the-art results, boosting the best published Inception score from 36.8 to 52.52 and reducing Frechet Inception distance from 27.62 to 18.65 on the challenging ImageNet dataset. Visualization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape.
* [[BEGAN](https://arxiv.org/abs/1703.10717)]
    [[pdf](https://arxiv.org/pdf/1703.10717.pdf)]
    * Title: BEGAN: Boundary Equilibrium Generative Adversarial Networks
    * Year: 31 Mar `2017`
    * Authors: David Berthelot, Thomas Schumm, Luke Metz
    * Abstract: We propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based Generative Adversarial Networks. This method balances the generator and discriminator during training. Additionally, it provides a new approximate convergence measure, fast and stable training and high visual quality. We also derive a way of controlling the trade-off between image diversity and visual quality. We focus on the image generation task, setting a new milestone in visual quality, even at higher resolutions. This is achieved while using a relatively simple model architecture and a standard training procedure.
* [[Towards the Automatic Anime Characters Creation with Generative Adversarial Networks](https://arxiv.org/abs/1708.05509)]
    [[pdf](https://arxiv.org/pdf/1708.05509.pdf)]
    * Title: Towards the Automatic Anime Characters Creation with Generative Adversarial Networks
    * Year: 18 Aug `2017`
    * Authors: Yanghua Jin, Jiakai Zhang, Minjun Li, Yingtao Tian, Huachun Zhu, Zhihao Fang
    * Abstract: Automatic generation of facial images has been well studied after the Generative Adversarial Network (GAN) came out. There exists some attempts applying the GAN model to the problem of generating facial images of anime characters, but none of the existing work gives a promising result. In this work, we explore the training of GAN models specialized on an anime facial image dataset. We address the issue from both the data and the model aspect, by collecting a more clean, well-suited dataset and leverage proper, empirical application of DRAGAN. With quantitative analysis and case studies we demonstrate that our efforts lead to a stable and high-quality model. Moreover, to assist people with anime character design, we build a website (http://make.girls.moe) with our pre-trained model available online, which makes the model easily accessible to general public.

## StyleGAN Family

* [[ProgressiveGAN](https://arxiv.org/abs/1710.10196)]
    [[pdf](https://arxiv.org/pdf/1710.10196.pdf)]
    * Title: Progressive Growing of GANs for Improved Quality, Stability, and Variation
    * Year: 27 Oct `2017`
    * Authors: Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
    * Abstract: We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024^2. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CelebA dataset.
* [[StyleGANv1](https://arxiv.org/abs/1812.04948)]
    [[pdf](https://arxiv.org/pdf/1812.04948.pdf)]
    * Title: A Style-Based Generator Architecture for Generative Adversarial Networks
    * Year: 12 Dec `2018`
    * Authors: Tero Karras, Samuli Laine, Timo Aila
    * Abstract: We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.
* [[StyleGANv2](https://arxiv.org/abs/1912.04958)]
    * Title: Analyzing and Improving the Image Quality of StyleGAN
    * Year: 03 Dec `2019`
    * Author: Tero Karras
    * Abstract: The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven unconditional generative image modeling. We expose and analyze several of its characteristic artifacts, and propose changes in both model architecture and training methods to address them. In particular, we redesign the generator normalization, revisit progressive growing, and regularize the generator to encourage good conditioning in the mapping from latent codes to images. In addition to improving image quality, this path length regularizer yields the additional benefit that the generator becomes significantly easier to invert. This makes it possible to reliably attribute a generated image to a particular network. We furthermore visualize how well the generator utilizes its output resolution, and identify a capacity problem, motivating us to train larger models for additional quality improvements. Overall, our improved model redefines the state of the art in unconditional image modeling, both in terms of existing distribution quality metrics as well as perceived image quality.
* [[StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273)]
    [[pdf](https://arxiv.org/pdf/2202.00273.pdf)]
    * Title: StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets
    * Year: 01 Feb `2022`
    * Authors: Axel Sauer, Katja Schwarz, Andreas Geiger
    * Abstract: Computer graphics has experienced a recent surge of data-centric approaches for photorealistic and controllable content creation. StyleGAN in particular sets new standards for generative modeling regarding image quality and controllability. However, StyleGAN's performance severely degrades on large unstructured datasets such as ImageNet. StyleGAN was designed for controllability; hence, prior works suspect its restrictive design to be unsuitable for diverse datasets. In contrast, we find the main limiting factor to be the current training strategy. Following the recently introduced Projected GAN paradigm, we leverage powerful neural network priors and a progressive growing strategy to successfully train the latest StyleGAN3 generator on ImageNet. Our final model, StyleGAN-XL, sets a new state-of-the-art on large-scale image synthesis and is the first to generate images at a resolution of 10242 at such a dataset scale. We demonstrate that this model can invert and edit images beyond the narrow domain of portraits or specific object classes.

## Variational Autoencoder Related

* [[An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)]
    [[pdf](https://arxiv.org/pdf/1906.02691.pdf)]
    * Title: An Introduction to Variational Autoencoders
    * Year: 06 Jun `2019`
    * Authors: Diederik P. Kingma, Max Welling
    * Abstract: Variational autoencoders provide a principled framework for learning deep latent-variable models and corresponding inference models. In this work, we provide an introduction to variational autoencoders and some important extensions.
* [[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)]
    [[pdf](https://arxiv.org/pdf/1312.6114.pdf)]
    * Title: Auto-Encoding Variational Bayes
    * Year: 20 Dec `2013`
    * Authors: Diederik P Kingma, Max Welling
    * Abstract: How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions is two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results.
* [[A Contrastive Learning Approach for Training Variational Autoencoder Priors](https://arxiv.org/abs/2010.02917)]
    [[pdf](https://arxiv.org/pdf/2010.02917.pdf)]
    * Title: A Contrastive Learning Approach for Training Variational Autoencoder Priors
    * Year: 06 Oct `2020`
    * Authors: Jyoti Aneja, Alexander Schwing, Jan Kautz, Arash Vahdat
    * Abstract: Variational autoencoders (VAEs) are one of the powerful likelihood-based generative models with applications in many domains. However, they struggle to generate high-quality images, especially when samples are obtained from the prior without any tempering. One explanation for VAEs' poor generative quality is the prior hole problem: the prior distribution fails to match the aggregate approximate posterior. Due to this mismatch, there exist areas in the latent space with high density under the prior that do not correspond to any encoded image. Samples from those areas are decoded to corrupted images. To tackle this issue, we propose an energy-based prior defined by the product of a base prior distribution and a reweighting factor, designed to bring the base closer to the aggregate posterior. We train the reweighting factor by noise contrastive estimation, and we generalize it to hierarchical VAEs with many latent variable groups. Our experiments confirm that the proposed noise contrastive priors improve the generative performance of state-of-the-art VAEs by a large margin on the MNIST, CIFAR-10, CelebA 64, and CelebA HQ 256 datasets. Our method is simple and can be applied to a wide variety of VAEs to improve the expressivity of their prior distribution.
* [[Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images](https://arxiv.org/abs/2011.10650)]
    [[pdf](https://arxiv.org/pdf/2011.10650.pdf)]
    * Title: Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images
    * Year: 20 Nov `2020`
    * Authors: Rewon Child
    * Abstract: We present a hierarchical VAE that, for the first time, generates samples quickly while outperforming the PixelCNN in log-likelihood on all natural image benchmarks. We begin by observing that, in theory, VAEs can actually represent autoregressive models, as well as faster, better models if they exist, when made sufficiently deep. Despite this, autoregressive models have historically outperformed VAEs in log-likelihood. We test if insufficient depth explains why by scaling a VAE to greater stochastic depth than previously explored and evaluating it CIFAR-10, ImageNet, and FFHQ. In comparison to the PixelCNN, these very deep VAEs achieve higher likelihoods, use fewer parameters, generate samples thousands of times faster, and are more easily applied to high-resolution images. Qualitative studies suggest this is because the VAE learns efficient hierarchical visual representations. We release our source code and models at this https URL.

## Wasserstein GAN Related

* [[Wasserstein GAN](https://arxiv.org/abs/1701.07875)]
    [[pdf](https://arxiv.org/pdf/1701.07875.pdf)]
    * Title: Wasserstein GAN
    * Year: 26 Jan `2017`
    * Authors: Martin Arjovsky, Soumith Chintala, Léon Bottou
    * Abstract: We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.
* [[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)]
    [[pdf](https://arxiv.org/pdf/1704.00028.pdf)]
    * Title: Improved Training of Wasserstein GANs
    * Year: 31 Mar `2017`
    * Authors: Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
    * Abstract: Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.
* [[On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894)]
    [[pdf](https://arxiv.org/pdf/1709.08894.pdf)]
    * Title: On the regularization of Wasserstein GANs
    * Year: 26 Sep `2017`
    * Authors: Henning Petzka, Asja Fischer, Denis Lukovnicov
    * Abstract: Since their invention, generative adversarial networks (GANs) have become a popular approach for learning to model a distribution of real (unlabeled) data. Convergence problems during training are overcome by Wasserstein GANs which minimize the distance between the model and the empirical distribution in terms of a different metric, but thereby introduce a Lipschitz constraint into the optimization problem. A simple way to enforce the Lipschitz constraint on the class of functions, which can be modeled by the neural network, is weight clipping. It was proposed that training can be improved by instead augmenting the loss by a regularization term that penalizes the deviation of the gradient of the critic (as a function of the network's input) from one. We present theoretical arguments why using a weaker regularization term enforcing the Lipschitz constraint is preferable. These arguments are supported by experimental results on toy data sets.

## Auto-Regressive Generative Model

* [[Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)]
    [[pdf](https://arxiv.org/pdf/1601.06759.pdf)]
    * Title: Pixel Recurrent Neural Networks
    * Year: 25 Jan `2016`
    * Authors: Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu
    * Abstract: Modeling the distribution of natural images is a landmark problem in unsupervised learning. This task requires an image model that is at once expressive, tractable and scalable. We present a deep neural network that sequentially predicts the pixels in an image along the two spatial dimensions. Our method models the discrete probability of the raw pixel values and encodes the complete set of dependencies in the image. Architectural novelties include fast two-dimensional recurrent layers and an effective use of residual connections in deep recurrent networks. We achieve log-likelihood scores on natural images that are considerably better than the previous state of the art. Our main results also provide benchmarks on the diverse ImageNet dataset. Samples generated from the model appear crisp, varied and globally coherent.

## Adversarial Learning (EfficientNetV2, 2021)

* [PDA](https://arxiv.org/abs/1909.04839)
    * Title: PDA: Progressive Data Augmentation for General Robustness of Deep Neural Networks
    * Year: 11 Sep `2019`
    * Authors: Hang Yu, Aishan Liu, Xianglong Liu, Gengchao Li, Ping Luo, Ran Cheng, Jichen Yang, Chongzhi Zhang
    * Abstract: Adversarial images are designed to mislead deep neural networks (DNNs), attracting great attention in recent years. Although several defense strategies achieved encouraging robustness against adversarial samples, most of them fail to improve the robustness on common corruptions such as noise, blur, and weather/digital effects (e.g. frost, pixelate). To address this problem, we propose a simple yet effective method, named Progressive Data Augmentation (PDA), which enables general robustness of DNNs by progressively injecting diverse adversarial noises during training. In other words, DNNs trained with PDA are able to obtain more robustness against both adversarial attacks as well as common corruptions than the recent state-of-the-art methods. We also find that PDA is more efficient than prior arts and able to prevent accuracy drop on clean samples without being attacked. Furthermore, we theoretically show that PDA can control the perturbation bound and guarantee better generalization ability than existing work. Extensive experiments on many benchmarks such as CIFAR-10, SVHN, and ImageNet demonstrate that PDA significantly outperforms its counterparts in various experimental setups.

## Diffusion Models

* [[Diffusion-GAN: Training GANs with Diffusion](https://arxiv.org/abs/2206.02262)]
    [[pdf](https://arxiv.org/pdf/2206.02262.pdf)]
    * Title: Diffusion-GAN: Training GANs with Diffusion
    * Year: 05 Jun `2022`
    * Authors: Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, Mingyuan Zhou
    * Abstract: Generative adversarial networks (GANs) are challenging to train stably, and a promising remedy of injecting instance noise into the discriminator input has not been very effective in practice. In this paper, we propose Diffusion-GAN, a novel GAN framework that leverages a forward diffusion chain to generate Gaussian-mixture distributed instance noise. Diffusion-GAN consists of three components, including an adaptive diffusion process, a diffusion timestep-dependent discriminator, and a generator. Both the observed and generated data are diffused by the same adaptive diffusion process. At each diffusion timestep, there is a different noise-to-data ratio and the timestep-dependent discriminator learns to distinguish the diffused real data from the diffused generated data. The generator learns from the discriminator's feedback by backpropagating through the forward diffusion chain, whose length is adaptively adjusted to balance the noise and data levels. We theoretically show that the discriminator's timestep-dependent strategy gives consistent and helpful guidance to the generator, enabling it to match the true data distribution. We demonstrate the advantages of Diffusion-GAN over strong GAN baselines on various datasets, showing that it can produce more realistic images with higher stability and data efficiency than state-of-the-art GANs.

## Text-to-Image Models

* [[Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396)]
    [[pdf](https://arxiv.org/pdf/1605.05396.pdf)]
    * Title: Generative Adversarial Text to Image Synthesis
    * Year: 17 May `2016`
    * Authors: Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, Honglak Lee
    * Abstract: Automatic synthesis of realistic images from text would be interesting and useful, but current AI systems are still far from this goal. However, in recent years generic and powerful recurrent neural network architectures have been developed to learn discriminative text feature representations. Meanwhile, deep convolutional generative adversarial networks (GANs) have begun to generate highly compelling images of specific categories, such as faces, album covers, and room interiors. In this work, we develop a novel deep architecture and GAN formulation to effectively bridge these advances in text and image model- ing, translating visual concepts from characters to pixels. We demonstrate the capability of our model to generate plausible images of birds and flowers from detailed text descriptions.

## Unclassified

* [[Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423)]
    [[pdf](https://arxiv.org/pdf/2106.12423.pdf)]
    * Title: Alias-Free Generative Adversarial Networks
    * Year: 23 Jun `2021`
    * Authors: Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, Timo Aila
    * Abstract: We observe that despite their hierarchical convolutional nature, the synthesis process of typical generative adversarial networks depends on absolute pixel coordinates in an unhealthy manner. This manifests itself as, e.g., detail appearing to be glued to image coordinates instead of the surfaces of depicted objects. We trace the root cause to careless signal processing that causes aliasing in the generator network. Interpreting all signals in the network as continuous, we derive generally applicable, small architectural changes that guarantee that unwanted information cannot leak into the hierarchical synthesis process. The resulting networks match the FID of StyleGAN2 but differ dramatically in their internal representations, and they are fully equivariant to translation and rotation even at subpixel scales. Our results pave the way for generative models better suited for video and animation.
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
* [[Generalization and Equilibrium in Generative Adversarial Nets (GANs)](https://arxiv.org/abs/1703.00573)]
    [[pdf](https://arxiv.org/pdf/1703.00573.pdf)]
    * Title: Generalization and Equilibrium in Generative Adversarial Nets (GANs)
    * Year: 02 Mar `2017`
    * Authors: Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, Yi Zhang
    * Abstract: We show that training of generative adversarial network (GAN) may not have good generalization properties; e.g., training may appear successful but the trained distribution may be far from target distribution in standard metrics. However, generalization does occur for a weaker metric called neural net distance. It is also shown that an approximate pure equilibrium exists in the discriminator/generator game for a special class of generators with natural training objectives when generator capacity and training set sizes are moderate. This existence of equilibrium inspires MIX+GAN protocol, which can be combined with any existing GAN training, and empirically shown to improve some of them.
* [[Conditional GAN](https://arxiv.org/abs/1411.1784)]
    [[pdf](https://arxiv.org/pdf/1411.1784.pdf)]
    * Title: Conditional Generative Adversarial Nets
    * Year: 06 Nov `2014`
    * Authors: Mehdi Mirza, Simon Osindero
    * Abstract: Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.
* [[Attribute-Guided Face Generation Using Conditional CycleGAN](https://arxiv.org/abs/1705.09966)]
    [[pdf](https://arxiv.org/pdf/1705.09966.pdf)]
    * Title: Attribute-Guided Face Generation Using Conditional CycleGAN
    * Year: 28 May `2017`
    * Authors: Yongyi Lu, Yu-Wing Tai, Chi-Keung Tang
    * Abstract: We are interested in attribute-guided face generation: given a low-res face input image, an attribute vector that can be extracted from a high-res image (attribute image), our new method generates a high-res face image for the low-res input that satisfies the given attributes. To address this problem, we condition the CycleGAN and propose conditional CycleGAN, which is designed to 1) handle unpaired training data because the training low/high-res and high-res attribute images may not necessarily align with each other, and to 2) allow easy control of the appearance of the generated face via the input attributes. We demonstrate impressive results on the attribute-guided conditional CycleGAN, which can synthesize realistic face images with appearance easily controlled by user-supplied attributes (e.g., gender, makeup, hair color, eyeglasses). Using the attribute image as identity to produce the corresponding conditional vector and by incorporating a face verification network, the attribute-guided network becomes the identity-guided conditional CycleGAN which produces impressive and interesting results on identity transfer. We demonstrate three applications on identity-guided conditional CycleGAN: identity-preserving face superresolution, face swapping, and frontal face generation, which consistently show the advantage of our new method.
* [[Boundary-Seeking Generative Adversarial Networks](https://arxiv.org/abs/1702.08431)]
    [[pdf](https://arxiv.org/pdf/1702.08431.pdf)]
    * Title: Boundary-Seeking Generative Adversarial Networks
    * Year: 27 Feb `2017`
    * Authors: R Devon Hjelm, Athul Paul Jacob, Tong Che, Adam Trischler, Kyunghyun Cho, Yoshua Bengio
    * Abstract: Generative adversarial networks (GANs) are a learning framework that rely on training a discriminator to estimate a measure of difference between a target and generated distributions. GANs, as normally formulated, rely on the generated samples being completely differentiable w.r.t. the generative parameters, and thus do not work for discrete data. We introduce a method for training GANs with discrete data that uses the estimated difference measure from the discriminator to compute importance weights for generated samples, thus providing a policy gradient for training the generator. The importance weights have a strong connection to the decision boundary of the discriminator, and we call our method boundary-seeking GANs (BGANs). We demonstrate the effectiveness of the proposed algorithm with discrete image and character-based natural language generation. In addition, the boundary-seeking objective extends to continuous data, which can be used to improve stability of training, and we demonstrate this on Celeba, Large-scale Scene Understanding (LSUN) bedrooms, and Imagenet without conditioning.
* [[Bayesian GAN](https://arxiv.org/abs/1705.09558)]
    [[pdf](https://arxiv.org/pdf/1705.09558.pdf)]
    * Title: Bayesian GAN
    * Year: 26 May `2017`
    * Authors: Yunus Saatchi, Andrew Gordon Wilson
    * Abstract: Generative adversarial networks (GANs) can implicitly learn rich distributions over images, audio, and data which are hard to model with an explicit likelihood. We present a practical Bayesian formulation for unsupervised and semi-supervised learning with GANs. Within this framework, we use stochastic gradient Hamiltonian Monte Carlo to marginalize the weights of the generator and discriminator networks. The resulting approach is straightforward and obtains good performance without any standard interventions such as feature matching, or mini-batch discrimination. By exploring an expressive posterior over the parameters of the generator, the Bayesian GAN avoids mode-collapse, produces interpretable and diverse candidate samples, and provides state-of-the-art quantitative results for semi-supervised learning on benchmarks including SVHN, CelebA, and CIFAR-10, outperforming DCGAN, Wasserstein GANs, and DCGAN ensembles.
* [[Bayesian Conditional Generative Adverserial Networks](https://arxiv.org/abs/1706.05477)]
    [[pdf](https://arxiv.org/pdf/1706.05477.pdf)]
    * Title: Bayesian Conditional Generative Adverserial Networks
    * Year: 17 Jun `2017`
    * Authors: M. Ehsan Abbasnejad, Qinfeng Shi, Iman Abbasnejad, Anton van den Hengel, Anthony Dick
    * Abstract: Traditional GANs use a deterministic generator function (typically a neural network) to transform a random noise input z to a sample x that the discriminator seeks to distinguish. We propose a new GAN called Bayesian Conditional Generative Adversarial Networks (BC-GANs) that use a random generator function to transform a deterministic input y′ to a sample x. Our BC-GANs extend traditional GANs to a Bayesian framework, and naturally handle unsupervised learning, supervised learning, and semi-supervised learning problems. Experiments show that the proposed BC-GANs outperforms the state-of-the-arts.
* [[APE-GAN](https://arxiv.org/abs/1707.05474)]
    [[pdf](https://arxiv.org/pdf/1707.05474.pdf)]
    * Title: APE-GAN: Adversarial Perturbation Elimination with GAN
    * Year: 18 Jul `2017`
    * Authors: Shiwei Shen, Guoqing Jin, Ke Gao, Yongdong Zhang
    * Abstract: Although neural networks could achieve state-of-the-art performance while recongnizing images, they often suffer a tremendous defeat from adversarial examples--inputs generated by utilizing imperceptible but intentional perturbation to clean samples from the datasets. How to defense against adversarial examples is an important problem which is well worth researching. So far, very few methods have provided a significant defense to adversarial examples. In this paper, a novel idea is proposed and an effective framework based Generative Adversarial Nets named APE-GAN is implemented to defense against the adversarial examples. The experimental results on three benchmark datasets including MNIST, CIFAR10 and ImageNet indicate that APE-GAN is effective to resist adversarial examples generated from five attacks.
* [[It Takes (Only) Two: Adversarial Generator-Encoder Networks](https://arxiv.org/abs/1704.02304)]
    [[pdf](https://arxiv.org/pdf/1704.02304.pdf)]
    * Title: It Takes (Only) Two: Adversarial Generator-Encoder Networks
    * Year: 07 Apr `2017`
    * Authors: Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
    * Abstract: We present a new autoencoder-type architecture that is trainable in an unsupervised mode, sustains both generation and inference, and has the quality of conditional and unconditional samples boosted by adversarial learning. Unlike previous hybrids of autoencoders and adversarial networks, the adversarial game in our approach is set up directly between the encoder and the generator, and no external mappings are trained in the process of learning. The game objective compares the divergences of each of the real and the generated data distributions with the prior distribution in the latent space. We show that direct generator-vs-encoder game leads to a tight coupling of the two components, resulting in samples and reconstructions of a comparable quality to some recently-proposed more complex architectures.
* [[Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464)]
    [[pdf](https://arxiv.org/pdf/1702.05464.pdf)]
    * Title: Adversarial Discriminative Domain Adaptation
    * Year: 17 Feb `2017`
    * Authors: Eric Tzeng, Judy Hoffman, Kate Saenko, Trevor Darrell
    * Abstract: Adversarial learning methods are a promising approach to training robust deep networks, and can generate complex samples across diverse domains. They also can improve recognition despite the presence of domain shift or dataset bias: several adversarial approaches to unsupervised domain adaptation have recently been introduced, which reduce the difference between the training and test domain distributions and thus improve generalization performance. Prior generative approaches show compelling visualizations, but are not optimal on discriminative tasks and can be limited to smaller shifts. Prior discriminative approaches could handle larger domain shifts, but imposed tied weights on the model and did not exploit a GAN-based loss. We first outline a novel generalized framework for adversarial adaptation, which subsumes recent state-of-the-art approaches as special cases, and we use this generalized view to better relate the prior approaches. We propose a previously unexplored instance of our general framework which combines discriminative modeling, untied weight sharing, and a GAN loss, which we call Adversarial Discriminative Domain Adaptation (ADDA). We show that ADDA is more effective yet considerably simpler than competing domain-adversarial methods, and demonstrate the promise of our approach by exceeding state-of-the-art unsupervised adaptation results on standard cross-domain digit classification tasks and a new more difficult cross-modality object classification task.
* [[Adversarial Autoencoders](https://arxiv.org/abs/1511.05644)]
    [[pdf](https://arxiv.org/pdf/1511.05644.pdf)]
    * Title: Adversarial Autoencoders
    * Year: 18 Nov `2015`
    * Authors: Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey
    * Abstract: In this paper, we propose the "adversarial autoencoder" (AAE), which is a probabilistic autoencoder that uses the recently proposed generative adversarial networks (GAN) to perform variational inference by matching the aggregated posterior of the hidden code vector of the autoencoder with an arbitrary prior distribution. Matching the aggregated posterior to the prior ensures that generating from any part of prior space results in meaningful samples. As a result, the decoder of the adversarial autoencoder learns a deep generative model that maps the imposed prior to the data distribution. We show how the adversarial autoencoder can be used in applications such as semi-supervised classification, disentangling style and content of images, unsupervised clustering, dimensionality reduction and data visualization. We performed experiments on MNIST, Street View House Numbers and Toronto Face datasets and show that adversarial autoencoders achieve competitive results in generative modeling and semi-supervised classification tasks.
* [[AdaGAN](https://arxiv.org/abs/1701.02386)]
    [[pdf](https://arxiv.org/pdf/1701.02386.pdf)]
    * Title: AdaGAN: Boosting Generative Models
    * Year: 09 Jan `2017`
    * Authors: Ilya Tolstikhin, Sylvain Gelly, Olivier Bousquet, Carl-Johann Simon-Gabriel, Bernhard Schölkopf
    * Abstract: Generative Adversarial Networks (GAN) (Goodfellow et al., 2014) are an effective method for training generative models of complex data such as natural images. However, they are notoriously hard to train and can suffer from the problem of missing modes where the model is not able to produce examples in certain regions of the space. We propose an iterative procedure, called AdaGAN, where at every step we add a new component into a mixture model by running a GAN algorithm on a reweighted sample. This is inspired by boosting algorithms, where many potentially weak individual predictors are greedily aggregated to form a strong composite predictor. We prove that such an incremental procedure leads to convergence to the true distribution in a finite number of steps if each step is optimal, and convergence at an exponential rate otherwise. We also illustrate experimentally that this procedure addresses the problem of missing modes.
* [[Activation Maximization Generative Adversarial Nets](https://arxiv.org/abs/1703.02000)]
    [[pdf](https://arxiv.org/pdf/1703.02000.pdf)]
    * Title: Activation Maximization Generative Adversarial Nets
    * Year: 06 Mar `2017`
    * Authors: Zhiming Zhou, Han Cai, Shu Rong, Yuxuan Song, Kan Ren, Weinan Zhang, Yong Yu, Jun Wang
    * Abstract: Class labels have been empirically shown useful in improving the sample quality of generative adversarial nets (GANs). In this paper, we mathematically study the properties of the current variants of GANs that make use of class label information. With class aware gradient and cross-entropy decomposition, we reveal how class labels and associated losses influence GAN's training. Based on that, we propose Activation Maximization Generative Adversarial Networks (AM-GAN) as an advanced solution. Comprehensive experiments have been conducted to validate our analysis and evaluate the effectiveness of our solution, where AM-GAN outperforms other strong baselines and achieves state-of-the-art Inception Score (8.91) on CIFAR-10. In addition, we demonstrate that, with the Inception ImageNet classifier, Inception Score mainly tracks the diversity of the generator, and there is, however, no reliable evidence that it can reflect the true sample quality. We thus propose a new metric, called AM Score, to provide a more accurate estimation of the sample quality. Our proposed model also outperforms the baseline methods in the new metric.
* [[Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)]
    [[pdf](https://arxiv.org/pdf/1611.04076.pdf)]
    * Title: Least Squares Generative Adversarial Networks
    * Year: 13 Nov `2016`
    * Authors: Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley
    * Abstract: Unsupervised learning with generative adversarial networks (GANs) has proven hugely successful. Regular GANs hypothesize the discriminator as a classifier with the sigmoid cross entropy loss function. However, we found that this loss function may lead to the vanishing gradients problem during the learning process. To overcome such a problem, we propose in this paper the Least Squares Generative Adversarial Networks (LSGANs) which adopt the least squares loss function for the discriminator. We show that minimizing the objective function of LSGAN yields minimizing the Pearson \chi^2 divergence. There are two benefits of LSGANs over regular GANs. First, LSGANs are able to generate higher quality images than regular GANs. Second, LSGANs perform more stable during the learning process. We evaluate LSGANs on five scene datasets and the experimental results show that the images generated by LSGANs are of better quality than the ones generated by regular GANs. We also conduct two comparison experiments between LSGANs and regular GANs to illustrate the stability of LSGANs.
* [[Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)]
    [[pdf](https://arxiv.org/pdf/1703.05192.pdf)]
    * Title: Learning to Discover Cross-Domain Relations with Generative Adversarial Networks
    * Year: 15 Mar `2017`
    * Authors: Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, Jiwon Kim
    * Abstract: While humans easily recognize relations between data from different domains without any supervision, learning to automatically discover them is in general very challenging and needs many ground-truth pairs that illustrate the relations. To avoid costly pairing, we address the task of discovering cross-domain relations given unpaired data. We propose a method based on generative adversarial networks that learns to discover relations between different domains (DiscoGAN). Using the discovered relations, our proposed network successfully transfers style from one domain to another while preserving key attributes such as orientation and face identity. Source code for official implementation is publicly available this https URL
* [[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)]
    [[pdf](https://arxiv.org/pdf/1411.1784.pdf)]
    * Title: Conditional Generative Adversarial Nets
    * Year: 06 Nov `2014`
    * Authors: Mehdi Mirza, Simon Osindero
    * Abstract: Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.
* [[Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)]
    [[pdf](https://arxiv.org/pdf/1711.10337.pdf)]
    * Title: Are GANs Created Equal? A Large-Scale Study
    * Year: 28 Nov `2017`
    * Authors: Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, Olivier Bousquet
    * Abstract: Generative adversarial networks (GAN) are a powerful subclass of generative models. Despite a very rich research activity leading to numerous interesting GAN algorithms, it is still very hard to assess which algorithm(s) perform better than others. We conduct a neutral, multi-faceted large-scale empirical study on state-of-the art models and evaluation measures. We find that most models can reach similar scores with enough hyperparameter optimization and random restarts. This suggests that improvements can arise from a higher computational budget and tuning more than fundamental algorithmic changes. To overcome some limitations of the current metrics, we also propose several data sets on which precision and recall can be computed. Our experimental results suggest that future GAN research should be based on more systematic and objective evaluation procedures. Finally, we did not find evidence that any of the tested algorithms consistently outperforms the non-saturating GAN introduced in \cite{goodfellow2014generative}.
* [[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)]
    [[pdf](https://arxiv.org/pdf/1508.06576.pdf)]
    * Title: A Neural Algorithm of Artistic Style
    * Year: 26 Aug `2015`
    * Authors: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
    * Abstract: In fine art, especially painting, humans have mastered the skill to create unique visual experiences through composing a complex interplay between the content and style of an image. Thus far the algorithmic basis of this process is unknown and there exists no artificial system with similar capabilities. However, in other key areas of visual perception such as object and face recognition near-human performance was recently demonstrated by a class of biologically inspired vision models called Deep Neural Networks. Here we introduce an artificial system based on a Deep Neural Network that creates artistic images of high perceptual quality. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images. Moreover, in light of the striking similarities between performance-optimised artificial neural networks and biological vision, our work offers a path forward to an algorithmic understanding of how humans create and perceive artistic imagery.
