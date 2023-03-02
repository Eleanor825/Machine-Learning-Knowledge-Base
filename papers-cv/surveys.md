# [Papers] Surveys <!-- omit in toc -->

count=14

## Table of Contents <!-- omit in toc -->

- [1. Deep Learning Approaches](#1-deep-learning-approaches)
- [2. Generative Adversarial Networks](#2-generative-adversarial-networks)
- [3. Transformers](#3-transformers)
- [4. Diffusion](#4-diffusion)
- [5. Others](#5-others)
- [6. SLAM](#6-slam)
  - [6.1. Classical Age (1986-2004)](#61-classical-age-1986-2004)
  - [6.2. Algorithmic Analysis Age (2004-2015)](#62-algorithmic-analysis-age-2004-2015)
  - [6.3. Robust Perception Age](#63-robust-perception-age)

## 1. Deep Learning Approaches

* [The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches](https://arxiv.org/abs/1803.01164)
    * Title: The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches
    * Year: 03 Mar `2018`
    * Authors: Md Zahangir Alom, Tarek M. Taha, Christopher Yakopcic, Stefan Westberg, Paheding Sidike, Mst Shamima Nasrin, Brian C Van Esesn, Abdul A S. Awwal, Vijayan K. Asari
    * Abstract: Deep learning has demonstrated tremendous success in variety of application domains in the past few years. This new field of machine learning has been growing rapidly and applied in most of the application domains with some new modalities of applications, which helps to open new opportunity. There are different methods have been proposed on different category of learning approaches, which includes supervised, semi-supervised and un-supervised learning. The experimental results show state-of-the-art performance of deep learning over traditional machine learning approaches in the field of Image Processing, Computer Vision, Speech Recognition, Machine Translation, Art, Medical imaging, Medical information processing, Robotics and control, Bio-informatics, Natural Language Processing (NLP), Cyber security, and many more. This report presents a brief survey on development of DL approaches, including Deep Neural Network (DNN), Convolutional Neural Network (CNN), Recurrent Neural Network (RNN) including Long Short Term Memory (LSTM) and Gated Recurrent Units (GRU), Auto-Encoder (AE), Deep Belief Network (DBN), Generative Adversarial Network (GAN), and Deep Reinforcement Learning (DRL). In addition, we have included recent development of proposed advanced variant DL techniques based on the mentioned DL approaches. Furthermore, DL approaches have explored and evaluated in different application domains are also included in this survey. We have also comprised recently developed frameworks, SDKs, and benchmark datasets that are used for implementing and evaluating deep learning approaches. There are some surveys have published on Deep Learning in Neural Networks [1, 38] and a survey on RL [234]. However, those papers have not discussed the individual advanced techniques for training large scale deep learning models and the recently developed method of generative models [1].

## 2. Generative Adversarial Networks

* [[Generative Adversarial Networks: An Overview](https://arxiv.org/abs/1710.07035)]
    * Title: Generative Adversarial Networks: An Overview
    * Year: 19 Oct `2017`
    * Authors: Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta, Anil A Bharath
    * Institutions: 
    * Abstract: Generative adversarial networks (GANs) provide a way to learn deep representations without extensively annotated training data. They achieve this through deriving backpropagation signals through a competitive process involving a pair of networks. The representations that can be learned by GANs may be used in a variety of applications, including image synthesis, semantic image editing, style transfer, image super-resolution and classification. The aim of this review paper is to provide an overview of GANs for the signal processing community, drawing on familiar analogies and concepts where possible. In addition to identifying different methods for training and constructing GANs, we also point to remaining challenges in their theory and application.

## 3. Transformers

* [[Recent Advances in Vision Transformer](https://arxiv.org/abs/2203.01536)]
    [[pdf](https://arxiv.org/pdf/2203.01536.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2203.01536/)]
    * Title: 
    * Year: 
    * Authors: Khawar Islam
* [[Transformers in Vision]()]
    [[pdf](https://arxiv.org/pdf/2101.01169.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2101.01169/)]

## 4. Diffusion

* [[Efficient Diffusion Models for Vision: A Survey](https://arxiv.org/abs/2210.09292)]
    [[pdf](https://arxiv.org/pdf/2210.09292.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2210.09292/)]
    * Title: Efficient Diffusion Models for Vision: A Survey
    * Year: 07 Oct `2022`
    * Authors: Anwaar Ulhaq, Naveed Akhtar, Ganna Pogrebna
    * Abstract: Diffusion Models (DMs) have demonstrated state-of-the-art performance in content generation without requiring adversarial training. These models are trained using a two-step process. First, a forward - diffusion - process gradually adds noise to a datum (usually an image). Then, a backward - reverse diffusion - process gradually removes the noise to turn it into a sample of the target distribution being modelled. DMs are inspired by non-equilibrium thermodynamics and have inherent high computational complexity. Due to the frequent function evaluations and gradient calculations in high-dimensional spaces, these models incur considerable computational overhead during both training and inference stages. This can not only preclude the democratization of diffusion-based modelling, but also hinder the adaption of diffusion models in real-life applications. Not to mention, the efficiency of computational models is fast becoming a significant concern due to excessive energy consumption and environmental scares. These factors have led to multiple contributions in the literature that focus on devising computationally efficient DMs. In this review, we present the most recent advances in diffusion models for vision, specifically focusing on the important design aspects that affect the computational efficiency of DMs. In particular, we emphasize the recently proposed design choices that have led to more efficient DMs. Unlike the other recent reviews, which discuss diffusion models from a broad perspective, this survey is aimed at pushing this research direction forward by highlighting the design strategies in the literature that are resulting in practicable models for the broader research community. We also provide a future outlook of diffusion models in vision from their computational efficiency viewpoint.
* [[Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796)]
    [[pdf](https://arxiv.org/pdf/2209.00796.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2209.00796/)]
    * Title: Diffusion Models: A Comprehensive Survey of Methods and Applications
    * Year: 02 Sep `2022`
    * Authors: Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Yingxia Shao, Wentao Zhang, Bin Cui, Ming-Hsuan Yang
    * Abstract: Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in many applications, including image synthesis, video generation, and molecule design. In this survey, we provide an overview of the rapidly expanding body of work on diffusion models, categorizing the research into three key areas: efficient sampling, improved likelihood estimation, and handling data with special structures. We also discuss the potential for combining diffusion models with other generative models for enhanced results. We further review the wide-ranging applications of diffusion models in fields spanning from computer vision, natural language processing, temporal data modeling, to interdisciplinary applications in other scientific disciplines. This survey aims to provide a contextualized, in-depth look at the state of diffusion models, identifying the key areas of focus and pointing to potential areas for further exploration. Github: this https URL.
* [[A Survey on Generative Diffusion Model](https://arxiv.org/abs/2209.02646)]
    [[pdf](https://arxiv.org/pdf/2209.02646.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2209.02646/)]
    * Title: A Survey on Generative Diffusion Model
    * Year: 06 Sep `2022`
    * Authors: Hanqun Cao, Cheng Tan, Zhangyang Gao, Guangyong Chen, Pheng-Ann Heng, Stan Z. Li
    * Abstract: Deep learning shows excellent potential in generation tasks thanks to deep latent representation. Generative models are classes of models that can generate observations randomly concerning certain implied parameters. Recently, the diffusion Model has become a rising class of generative models by its power-generating ability. Nowadays, great achievements have been reached. More applications except for computer vision, speech generation, bioinformatics, and natural language processing are to be explored in this field. However, the diffusion model has its genuine drawback of a slow generation process, single data types, low likelihood, and the inability for dimension reduction. They are leading to many enhanced works. This survey makes a summary of the field of the diffusion model. We first state the main problem with two landmark works -- DDPM and DSM, and a unified landmark work -- Score SDE. Then, we present improved techniques for existing problems in the diffusion-based model field, including speed-up improvement For model speed-up improvement, data structure diversification, likelihood optimization, and dimension reduction. Regarding existing models, we also provide a benchmark of FID score, IS, and NLL according to specific NFE. Moreover, applications with diffusion models are introduced including computer vision, sequence modeling, audio, and AI for science. Finally, there is a summarization of this field together with limitations \& further directions. The summation of existing well-classified methods is in our Github:this https URL.
* [[Diffusion Models in Vision: A Survey](https://arxiv.org/abs/2209.04747)]
    [[pdf](https://arxiv.org/pdf/2209.04747.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2209.04747/)]
    * Title: Diffusion Models in Vision: A Survey
    * Year: 10 Sep `2022`
    * Authors: Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Mubarak Shah
    * Abstract: Denoising diffusion models represent a recent emerging topic in computer vision, demonstrating remarkable results in the area of generative modeling. A diffusion model is a deep generative model that is based on two stages, a forward diffusion stage and a reverse diffusion stage. In the forward diffusion stage, the input data is gradually perturbed over several steps by adding Gaussian noise. In the reverse stage, a model is tasked at recovering the original input data by learning to gradually reverse the diffusion process, step by step. Diffusion models are widely appreciated for the quality and diversity of the generated samples, despite their known computational burdens, i.e. low speeds due to the high number of steps involved during sampling. In this survey, we provide a comprehensive review of articles on denoising diffusion models applied in vision, comprising both theoretical and practical contributions in the field. First, we identify and present three generic diffusion modeling frameworks, which are based on denoising diffusion probabilistic models, noise conditioned score networks, and stochastic differential equations. We further discuss the relations between diffusion models and other deep generative models, including variational auto-encoders, generative adversarial networks, energy-based models, autoregressive models and normalizing flows. Then, we introduce a multi-perspective categorization of diffusion models applied in computer vision. Finally, we illustrate the current limitations of diffusion models and envision some interesting directions for future research.
* [[Diffusion Models for Medical Image Analysis: A Comprehensive Survey](https://arxiv.org/abs/2211.07804)]
    [[pdf](https://arxiv.org/pdf/2211.07804.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2211.07804/)]
    * Title: Diffusion Models for Medical Image Analysis: A Comprehensive Survey
    * Year: 14 Nov `2022`
    * Authors: Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Moein Heidari, Reza Azad, Mohsen Fayyaz, Ilker Hacihaliloglu, Dorit Merhof
    * Abstract: Denoising diffusion models, a class of generative models, have garnered immense interest lately in various deep-learning problems. A diffusion probabilistic model defines a forward diffusion stage where the input data is gradually perturbed over several steps by adding Gaussian noise and then learns to reverse the diffusion process to retrieve the desired noise-free data from noisy data samples. Diffusion models are widely appreciated for their strong mode coverage and quality of the generated samples despite their known computational burdens. Capitalizing on the advances in computer vision, the field of medical imaging has also observed a growing interest in diffusion models. To help the researcher navigate this profusion, this survey intends to provide a comprehensive overview of diffusion models in the discipline of medical image analysis. Specifically, we introduce the solid theoretical foundation and fundamental concepts behind diffusion models and the three generic diffusion modelling frameworks: diffusion probabilistic models, noise-conditioned score networks, and stochastic differential equations. Then, we provide a systematic taxonomy of diffusion models in the medical domain and propose a multi-perspective categorization based on their application, imaging modality, organ of interest, and algorithms. To this end, we cover extensive applications of diffusion models in the medical domain. Furthermore, we emphasize the practical use case of some selected approaches, and then we discuss the limitations of the diffusion models in the medical domain and propose several directions to fulfill the demands of this field. Finally, we gather the overviewed studies with their available open-source implementations at this https URL.

## 5. Others

* [Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538)
    * Title: Representation Learning: A Review and New Perspectives
    * Year: 24 Jun `2012`
    * Author: Yoshua Bengio
    * Abstract: The success of machine learning algorithms generally depends on data representation, and we hypothesize that this is because different representations can entangle and hide more or less the different explanatory factors of variation behind the data. Although specific domain knowledge can be used to help design representations, learning with generic priors can also be used, and the quest for AI is motivating the design of more powerful representation-learning algorithms implementing such priors. This paper reviews recent work in the area of unsupervised feature learning and deep learning, covering advances in probabilistic models, auto-encoders, manifold learning, and deep networks. This motivates longer-term unanswered questions about the appropriate objectives for learning good representations, for computing representations (i.e., inference), and the geometrical connections between representation learning, density estimation and manifold learning.

## 6. SLAM

### 6.1. Classical Age (1986-2004)

* [Simultaneous localization and mapping: part I](https://ieeexplore.ieee.org/document/1638022)
    * Title: Simultaneous localization and mapping: part I
    * Year: `2006`
    * Author: H. Durrant-Whyte
    * Abstract: This paper describes the simultaneous localization and mapping (SLAM) problem and the essential methods for solving the SLAM problem and summarizes key implementations and demonstrations of the method. While there are still many practical issues to overcome, especially in more complex outdoor environments, the general SLAM method is now a well understood and established part of robotics. Another part of the tutorial summarized more recent works in addressing some of the remaining issues in SLAM, including computation, feature representation, and data association.
* [Simultaneous localization and mapping (SLAM): part II](https://ieeexplore.ieee.org/document/1678144)
    * Title: Simultaneous localization and mapping (SLAM): part II
    * Year: `2006`
    * Author: T. Bailey
    * Abstract: This paper discusses the recursive Bayesian formulation of the simultaneous localization and mapping (SLAM) problem in which probability distributions or estimates of absolute or relative locations of landmarks and vehicle pose are obtained. The paper focuses on three key areas: computational complexity; data association; and environment representation.

### 6.2. Algorithmic Analysis Age (2004-2015)

* [A review of recent developments in Simultaneous Localization and Mapping](https://ieeexplore.ieee.org/abstract/document/6038117)
    * Title: A review of recent developments in Simultaneous Localization and Mapping
    * Year: `2011`
    * Author: Gamini Dissanayake
    * Abstract: Simultaneous Localization and Mapping (SLAM) problem has been an active area of research in robotics for more than a decade. Many fundamental and practical aspects of SLAM have been addressed and some impressive practical solutions have been demonstrated. The aim of this paper is to provide a review of the current state of the research on feature based SLAM, in particular to examine the current understanding of the fundamental properties of the SLAM problem and associated issues with the view to consolidate recent achievements.
* [Visual Place Recognition: A Survey](https://ieeexplore.ieee.org/document/7339473)
    * Title: Visual Place Recognition: A Survey
    * Year: `2016`
    * Author: Stephanie Lowry
    * Abstract: Visual place recognition is a challenging problem due to the vast range of ways in which the appearance of real-world places can vary. In recent years, improvements in visual sensing capabilities, an ever-increasing focus on long-term mobile robot autonomy, and the ability to draw on state-of-the-art research in other disciplines-particularly recognition in computer vision and animal navigation in neuroscience-have all contributed to significant advances in visual place recognition systems. This paper presents a survey of the visual place recognition research landscape. We start by introducing the concepts behind place recognition-the role of place recognition in the animal kingdom, how a “place” is defined in a robotics context, and the major components of a place recognition system. Long-term robot operations have revealed that changing appearance can be a significant factor in visual place recognition failure; therefore, we discuss how place recognition solutions can implicitly or explicitly account for appearance change within the environment. Finally, we close with a discussion on the future of visual place recognition, in particular with respect to the rapid advances being made in the related fields of deep learning, semantic scene understanding, and video description.

### 6.3. Robust Perception Age

* [Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age](https://arxiv.org/abs/1606.05830)
    * Title: Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age
    * Year: 19 Jun `2016`
    * Authors: Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Latif, Davide Scaramuzza, Jose Neira, Ian Reid, John J. Leonard
    * Abstract: Simultaneous Localization and Mapping (SLAM)consists in the concurrent construction of a model of the environment (the map), and the estimation of the state of the robot moving within it. The SLAM community has made astonishing progress over the last 30 years, enabling large-scale real-world applications, and witnessing a steady transition of this technology to industry. We survey the current state of SLAM. We start by presenting what is now the de-facto standard formulation for SLAM. We then review related work, covering a broad set of topics including robustness and scalability in long-term mapping, metric and semantic representations for mapping, theoretical performance guarantees, active SLAM and exploration, and other new frontiers. This paper simultaneously serves as a position paper and tutorial to those who are users of SLAM. By looking at the published research with a critical eye, we delineate open challenges and new research issues, that still deserve careful scientific investigation. The paper also contains the authors' take on two questions that often animate discussions during robotics conferences: Do robots need SLAM? and Is SLAM solved?
