* [[Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)]
    [[pdf](https://arxiv.org/pdf/1802.05957.pdf)]
    * Title: Spectral Normalization for Generative Adversarial Networks
    * Year: 16 Feb `2018`
    * Authors: Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    * Abstract: One of the challenges in the study of generative adversarial networks is the instability of its training. In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator. Our new normalization technique is computationally light and easy to incorporate into existing implementations. We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques.
* [[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)]
    [[pdf](https://arxiv.org/pdf/1805.08318.pdf)]
    * Title: Self-Attention Generative Adversarial Networks
    * Year: 21 May `2018`
    * Authors: Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena
    * Abstract: In this paper, we propose the Self-Attention Generative Adversarial Network (SAGAN) which allows attention-driven, long-range dependency modeling for image generation tasks. Traditional convolutional GANs generate high-resolution details as a function of only spatially local points in lower-resolution feature maps. In SAGAN, details can be generated using cues from all feature locations. Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other. Furthermore, recent work has shown that generator conditioning affects GAN performance. Leveraging this insight, we apply spectral normalization to the GAN generator and find that this improves training dynamics. The proposed SAGAN achieves the state-of-the-art results, boosting the best published Inception score from 36.8 to 52.52 and reducing Frechet Inception distance from 27.62 to 18.65 on the challenging ImageNet dataset. Visualization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape.
* [[BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717)]
    [[pdf](https://arxiv.org/pdf/1703.10717.pdf)]
    * Title: BEGAN: Boundary Equilibrium Generative Adversarial Networks
    * Year: 31 Mar `2017`
    * Authors: David Berthelot, Thomas Schumm, Luke Metz
    * Abstract: We propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based Generative Adversarial Networks. This method balances the generator and discriminator during training. Additionally, it provides a new approximate convergence measure, fast and stable training and high visual quality. We also derive a way of controlling the trade-off between image diversity and visual quality. We focus on the image generation task, setting a new milestone in visual quality, even at higher resolutions. This is achieved while using a relatively simple model architecture and a standard training procedure.
* [[An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)]
    [[pdf](https://arxiv.org/pdf/1906.02691.pdf)]
    * Title: An Introduction to Variational Autoencoders
    * Year: 06 Jun `2019`
    * Authors: Diederik P. Kingma, Max Welling
    * Abstract: Variational autoencoders provide a principled framework for learning deep latent-variable models and corresponding inference models. In this work, we provide an introduction to variational autoencoders and some important extensions.
* [[Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)]
    [[pdf](https://arxiv.org/pdf/1601.06759.pdf)]
    * Title: Pixel Recurrent Neural Networks
    * Year: 25 Jan `2016`
    * Authors: Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu
    * Abstract: Modeling the distribution of natural images is a landmark problem in unsupervised learning. This task requires an image model that is at once expressive, tractable and scalable. We present a deep neural network that sequentially predicts the pixels in an image along the two spatial dimensions. Our method models the discrete probability of the raw pixel values and encodes the complete set of dependencies in the image. Architectural novelties include fast two-dimensional recurrent layers and an effective use of residual connections in deep recurrent networks. We achieve log-likelihood scores on natural images that are considerably better than the previous state of the art. Our main results also provide benchmarks on the diverse ImageNet dataset. Samples generated from the model appear crisp, varied and globally coherent.
* [[Towards the Automatic Anime Characters Creation with Generative Adversarial Networks](https://arxiv.org/abs/1708.05509)]
    [[pdf](https://arxiv.org/pdf/1708.05509.pdf)]
    * Title: Towards the Automatic Anime Characters Creation with Generative Adversarial Networks
    * Year: 18 Aug `2017`
    * Authors: Yanghua Jin, Jiakai Zhang, Minjun Li, Yingtao Tian, Huachun Zhu, Zhihao Fang
    * Abstract: Automatic generation of facial images has been well studied after the Generative Adversarial Network (GAN) came out. There exists some attempts applying the GAN model to the problem of generating facial images of anime characters, but none of the existing work gives a promising result. In this work, we explore the training of GAN models specialized on an anime facial image dataset. We address the issue from both the data and the model aspect, by collecting a more clean, well-suited dataset and leverage proper, empirical application of DRAGAN. With quantitative analysis and case studies we demonstrate that our efforts lead to a stable and high-quality model. Moreover, to assist people with anime character design, we build a website (http://make.girls.moe) with our pre-trained model available online, which makes the model easily accessible to general public.
