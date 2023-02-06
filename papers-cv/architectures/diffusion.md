# [Papers][Vision] Diffusion Models <!-- omit in toc -->

count=28

## Table of Contents <!-- omit in toc -->

- [Surveys](#surveys)
- [Image Synthesis](#image-synthesis)
- [Image Editing](#image-editing)
- [Image Denoising](#image-denoising)
- [Unclassified](#unclassified)
  - [Mon Jan 23, 2023 Readings](#mon-jan-23-2023-readings)
  - [Mon Feb 06, 2023 Readings](#mon-feb-06-2023-readings)

----------------------------------------------------------------------------------------------------

## Surveys

* [[Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796)]
    [[pdf](https://arxiv.org/pdf/2209.00796.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2209.00796/)]
    * Title: Diffusion Models: A Comprehensive Survey of Methods and Applications
    * Year: 02 Sep `2022`
    * Authors: Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Yingxia Shao, Wentao Zhang, Bin Cui, Ming-Hsuan Yang
    * Abstract: Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in many applications, including image synthesis, video generation, and molecule design. In this survey, we provide an overview of the rapidly expanding body of work on diffusion models, categorizing the research into three key areas: efficient sampling, improved likelihood estimation, and handling data with special structures. We also discuss the potential for combining diffusion models with other generative models for enhanced results. We further review the wide-ranging applications of diffusion models in fields spanning from computer vision, natural language processing, temporal data modeling, to interdisciplinary applications in other scientific disciplines. This survey aims to provide a contextualized, in-depth look at the state of diffusion models, identifying the key areas of focus and pointing to potential areas for further exploration. Github: this https URL.
* [[Efficient Diffusion Models for Vision: A Survey](https://arxiv.org/abs/2210.09292)]
    [[pdf](https://arxiv.org/pdf/2210.09292.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2210.09292/)]
    * Title: Efficient Diffusion Models for Vision: A Survey
    * Year: 07 Oct `2022`
    * Authors: Anwaar Ulhaq, Naveed Akhtar, Ganna Pogrebna
    * Abstract: Diffusion Models (DMs) have demonstrated state-of-the-art performance in content generation without requiring adversarial training. These models are trained using a two-step process. First, a forward - diffusion - process gradually adds noise to a datum (usually an image). Then, a backward - reverse diffusion - process gradually removes the noise to turn it into a sample of the target distribution being modelled. DMs are inspired by non-equilibrium thermodynamics and have inherent high computational complexity. Due to the frequent function evaluations and gradient calculations in high-dimensional spaces, these models incur considerable computational overhead during both training and inference stages. This can not only preclude the democratization of diffusion-based modelling, but also hinder the adaption of diffusion models in real-life applications. Not to mention, the efficiency of computational models is fast becoming a significant concern due to excessive energy consumption and environmental scares. These factors have led to multiple contributions in the literature that focus on devising computationally efficient DMs. In this review, we present the most recent advances in diffusion models for vision, specifically focusing on the important design aspects that affect the computational efficiency of DMs. In particular, we emphasize the recently proposed design choices that have led to more efficient DMs. Unlike the other recent reviews, which discuss diffusion models from a broad perspective, this survey is aimed at pushing this research direction forward by highlighting the design strategies in the literature that are resulting in practicable models for the broader research community. We also provide a future outlook of diffusion models in vision from their computational efficiency viewpoint.
* [[Diffusion Models in Vision: A Survey](https://arxiv.org/abs/2209.04747)]
    [[pdf](https://arxiv.org/pdf/2209.04747.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2209.04747/)]
    * Title: Diffusion Models in Vision: A Survey
    * Year: 10 Sep `2022`
    * Authors: Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Mubarak Shah
    * Abstract: Denoising diffusion models represent a recent emerging topic in computer vision, demonstrating remarkable results in the area of generative modeling. A diffusion model is a deep generative model that is based on two stages, a forward diffusion stage and a reverse diffusion stage. In the forward diffusion stage, the input data is gradually perturbed over several steps by adding Gaussian noise. In the reverse stage, a model is tasked at recovering the original input data by learning to gradually reverse the diffusion process, step by step. Diffusion models are widely appreciated for the quality and diversity of the generated samples, despite their known computational burdens, i.e. low speeds due to the high number of steps involved during sampling. In this survey, we provide a comprehensive review of articles on denoising diffusion models applied in vision, comprising both theoretical and practical contributions in the field. First, we identify and present three generic diffusion modeling frameworks, which are based on denoising diffusion probabilistic models, noise conditioned score networks, and stochastic differential equations. We further discuss the relations between diffusion models and other deep generative models, including variational auto-encoders, generative adversarial networks, energy-based models, autoregressive models and normalizing flows. Then, we introduce a multi-perspective categorization of diffusion models applied in computer vision. Finally, we illustrate the current limitations of diffusion models and envision some interesting directions for future research.
* [[A Survey on Generative Diffusion Model](https://arxiv.org/abs/2209.02646)]
    [[pdf](https://arxiv.org/pdf/2209.02646.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2209.02646/)]
    * Title: A Survey on Generative Diffusion Model
    * Year: 06 Sep `2022`
    * Authors: Hanqun Cao, Cheng Tan, Zhangyang Gao, Guangyong Chen, Pheng-Ann Heng, Stan Z. Li
    * Abstract: Deep learning shows excellent potential in generation tasks thanks to deep latent representation. Generative models are classes of models that can generate observations randomly concerning certain implied parameters. Recently, the diffusion Model has become a rising class of generative models by its power-generating ability. Nowadays, great achievements have been reached. More applications except for computer vision, speech generation, bioinformatics, and natural language processing are to be explored in this field. However, the diffusion model has its genuine drawback of a slow generation process, single data types, low likelihood, and the inability for dimension reduction. They are leading to many enhanced works. This survey makes a summary of the field of the diffusion model. We first state the main problem with two landmark works -- DDPM and DSM, and a unified landmark work -- Score SDE. Then, we present improved techniques for existing problems in the diffusion-based model field, including speed-up improvement For model speed-up improvement, data structure diversification, likelihood optimization, and dimension reduction. Regarding existing models, we also provide a benchmark of FID score, IS, and NLL according to specific NFE. Moreover, applications with diffusion models are introduced including computer vision, sequence modeling, audio, and AI for science. Finally, there is a summarization of this field together with limitations \& further directions. The summation of existing well-classified methods is in our Github:this https URL.
* [[Diffusion Models for Medical Image Analysis: A Comprehensive Survey](https://arxiv.org/abs/2211.07804)]
    [[pdf](https://arxiv.org/pdf/2211.07804.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2211.07804/)]
    * Title: Diffusion Models for Medical Image Analysis: A Comprehensive Survey
    * Year: 14 Nov `2022`
    * Authors: Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Moein Heidari, Reza Azad, Mohsen Fayyaz, Ilker Hacihaliloglu, Dorit Merhof
    * Abstract: Denoising diffusion models, a class of generative models, have garnered immense interest lately in various deep-learning problems. A diffusion probabilistic model defines a forward diffusion stage where the input data is gradually perturbed over several steps by adding Gaussian noise and then learns to reverse the diffusion process to retrieve the desired noise-free data from noisy data samples. Diffusion models are widely appreciated for their strong mode coverage and quality of the generated samples despite their known computational burdens. Capitalizing on the advances in computer vision, the field of medical imaging has also observed a growing interest in diffusion models. To help the researcher navigate this profusion, this survey intends to provide a comprehensive overview of diffusion models in the discipline of medical image analysis. Specifically, we introduce the solid theoretical foundation and fundamental concepts behind diffusion models and the three generic diffusion modelling frameworks: diffusion probabilistic models, noise-conditioned score networks, and stochastic differential equations. Then, we provide a systematic taxonomy of diffusion models in the medical domain and propose a multi-perspective categorization based on their application, imaging modality, organ of interest, and algorithms. To this end, we cover extensive applications of diffusion models in the medical domain. Furthermore, we emphasize the practical use case of some selected approaches, and then we discuss the limitations of the diffusion models in the medical domain and propose several directions to fulfill the demands of this field. Finally, we gather the overviewed studies with their available open-source implementations at this https URL.

## Image Synthesis

* [[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)]
    [[pdf](https://arxiv.org/pdf/2112.10752.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2112.10752/)]
    * Title: High-Resolution Image Synthesis with Latent Diffusion Models
    * Year: 20 Dec `2021`
    * Authors: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
    * Abstract: By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs. Code is available at this https URL .
* [[More Control for Free! Image Synthesis with Semantic Diffusion Guidance](https://arxiv.org/abs/2112.05744)]
    [[pdf](https://arxiv.org/pdf/2112.05744.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2112.05744/)]
    * Title: More Control for Free! Image Synthesis with Semantic Diffusion Guidance
    * Year: 10 Dec `2021`
    * Authors: Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, Trevor Darrell
    * Abstract: Controllable image synthesis models allow creation of diverse images based on text instructions or guidance from a reference image. Recently, denoising diffusion probabilistic models have been shown to generate more realistic imagery than prior methods, and have been successfully demonstrated in unconditional and class-conditional settings. We investigate fine-grained, continuous control of this model class, and introduce a novel unified framework for semantic diffusion guidance, which allows either language or image guidance, or both. Guidance is injected into a pretrained unconditional diffusion model using the gradient of image-text or image matching scores, without re-training the diffusion model. We explore CLIP-based language guidance as well as both content and style-based image guidance in a unified framework. Our text-guided synthesis approach can be applied to datasets without associated text annotations. We conduct experiments on FFHQ and LSUN datasets, and show results on fine-grained text-guided image synthesis, synthesis of images related to a style or content reference image, and examples with both textual and image guidance.
* [[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)]
    [[pdf](https://arxiv.org/pdf/2105.05233.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2105.05233/)]
    * Title: Diffusion Models Beat GANs on Image Synthesis
    * Year: 11 May `2021`
    * Authors: Prafulla Dhariwal, Alex Nichol
    * Abstract: We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models. We achieve this on unconditional image synthesis by finding a better architecture through a series of ablations. For conditional image synthesis, we further improve sample quality with classifier guidance: a simple, compute-efficient method for trading off diversity for fidelity using gradients from a classifier. We achieve an FID of 2.97 on ImageNet 128$\times$128, 4.59 on ImageNet 256$\times$256, and 7.72 on ImageNet 512$\times$512, and we match BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution. Finally, we find that classifier guidance combines well with upsampling diffusion models, further improving FID to 3.94 on ImageNet 256$\times$256 and 3.85 on ImageNet 512$\times$512. We release our code at this https URL
* [[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)]
    [[pdf](https://arxiv.org/pdf/2212.09748.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2212.09748/)]
    * Title: Scalable Diffusion Models with Transformers
    * Year: 19 Dec `2022`
    * Authors: William Peebles, Saining Xie
    * Abstract: We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches. We analyze the scalability of our Diffusion Transformers (DiTs) through the lens of forward pass complexity as measured by Gflops. We find that DiTs with higher Gflops -- through increased transformer depth/width or increased number of input tokens -- consistently have lower FID. In addition to possessing good scalability properties, our largest DiT-XL/2 models outperform all prior diffusion models on the class-conditional ImageNet 512x512 and 256x256 benchmarks, achieving a state-of-the-art FID of 2.27 on the latter.
* [[Diffusion Models already have a Semantic Latent Space](https://arxiv.org/abs/2210.10960)]
    [[pdf](https://arxiv.org/pdf/2210.10960.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2210.10960/)]
    * Title: Diffusion Models already have a Semantic Latent Space
    * Year: 20 Oct `2022`
    * Authors: Mingi Kwon, Jaeseok Jeong, Youngjung Uh
    * Abstract: Diffusion models achieve outstanding generative performance in various domains. Despite their great success, they lack semantic latent space which is essential for controlling the generative process. To address the problem, we propose asymmetric reverse process (Asyrp) which discovers the semantic latent space in frozen pretrained diffusion models. Our semantic latent space, named h-space, has nice properties for accommodating semantic image manipulation: homogeneity, linearity, robustness, and consistency across timesteps. In addition, we introduce a principled design of the generative process for versatile editing and quality boost ing by quantifiable measures: editing strength of an interval and quality deficiency at a timestep. Our method is applicable to various architectures (DDPM++, iD- DPM, and ADM) and datasets (CelebA-HQ, AFHQ-dog, LSUN-church, LSUN- bedroom, and METFACES). Project page: this https URL
* [[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)]
    [[pdf](https://arxiv.org/pdf/2208.12242.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2208.12242/)]
    * Title: DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation
    * Year: 25 Aug `2022`
    * Authors: Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman
    * Abstract: Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for "personalization" of text-to-image diffusion models (specializing them to users' needs). Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model (Imagen, although our method is not limited to a specific model) such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can then be used to synthesize fully-novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, our technique enables synthesizing the subject in diverse scenes, poses, views, and lighting conditions that do not appear in the reference images. We apply our technique to several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, appearance modification, and artistic rendering (all while preserving the subject's key features). Project page: this https URL
* [[Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)]
    [[pdf](https://arxiv.org/pdf/2111.05826.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2111.05826/)]
    * Title: Palette: Image-to-Image Diffusion Models
    * Year: 10 Nov `2021`
    * Authors: Chitwan Saharia, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim Salimans, David J. Fleet, Mohammad Norouzi
    * Abstract: This paper develops a unified framework for image-to-image translation based on conditional diffusion models and evaluates this framework on four challenging image-to-image translation tasks, namely colorization, inpainting, uncropping, and JPEG restoration. Our simple implementation of image-to-image diffusion models outperforms strong GAN and regression baselines on all tasks, without task-specific hyper-parameter tuning, architecture customization, or any auxiliary loss or sophisticated new techniques needed. We uncover the impact of an L2 vs. L1 loss in the denoising diffusion objective on sample diversity, and demonstrate the importance of self-attention in the neural architecture through empirical studies. Importantly, we advocate a unified evaluation protocol based on ImageNet, with human evaluation and sample quality scores (FID, Inception Score, Classification Accuracy of a pre-trained ResNet-50, and Perceptual Distance against original images). We expect this standardized evaluation protocol to play a role in advancing image-to-image translation research. Finally, we show that a generalist, multi-task diffusion model performs as well or better than task-specific specialist counterparts. Check out this https URL for an overview of the results.
* [[Video Diffusion Models](https://arxiv.org/abs/2204.03458)]
    [[pdf](https://arxiv.org/pdf/2204.03458.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2204.03458/)]
    * Title: Video Diffusion Models
    * Year: 07 Apr `2022`
    * Authors: Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet
    * Abstract: Generating temporally coherent high fidelity video is an important milestone in generative modeling research. We make progress towards this milestone by proposing a diffusion model for video generation that shows very promising initial results. Our model is a natural extension of the standard image diffusion architecture, and it enables jointly training from image and video data, which we find to reduce the variance of minibatch gradients and speed up optimization. To generate long and higher resolution videos we introduce a new conditional sampling technique for spatial and temporal video extension that performs better than previously proposed methods. We present the first results on a large text-conditioned video generation task, as well as state-of-the-art results on established benchmarks for video prediction and unconditional video generation. Supplementary material is available at this https URL

## Image Editing

* [[Imagic](https://arxiv.org/abs/2210.09276)]
    [[pdf](https://arxiv.org/pdf/2210.09276.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2210.09276/)]
    * Title: Imagic: Text-Based Real Image Editing with Diffusion Models
    * Year: 17 Oct `2022`
    * Authors: Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, Michal Irani
    * Abstract: Text-conditioned image editing has recently attracted considerable interest. However, most methods are currently either limited to specific editing types (e.g., object overlay, style transfer), or apply to synthetically generated images, or require multiple input images of a common object. In this paper we demonstrate, for the very first time, the ability to apply complex (e.g., non-rigid) text-guided semantic edits to a single real image. For example, we can change the posture and composition of one or multiple objects inside an image, while preserving its original characteristics. Our method can make a standing dog sit down or jump, cause a bird to spread its wings, etc. -- each within its single high-resolution natural image provided by the user. Contrary to previous work, our proposed method requires only a single input image and a target text (the desired edit). It operates on real images, and does not require any additional inputs (such as image masks or additional views of the object). Our method, which we call "Imagic", leverages a pre-trained text-to-image diffusion model for this task. It produces a text embedding that aligns with both the input image and the target text, while fine-tuning the diffusion model to capture the image-specific appearance. We demonstrate the quality and versatility of our method on numerous inputs from various domains, showcasing a plethora of high quality complex semantic image edits, all within a single unified framework.
* [[Blended Diffusion for Text-driven Editing of Natural Images](https://arxiv.org/abs/2111.14818)]
    [[pdf](https://arxiv.org/pdf/2111.14818.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2111.14818/)]
    * Title: Blended Diffusion for Text-driven Editing of Natural Images
    * Year: 29 Nov `2021`
    * Authors: Omri Avrahami, Dani Lischinski, Ohad Fried
    * Abstract: Natural language offers a highly intuitive interface for image editing. In this paper, we introduce the first solution for performing local (region-based) edits in generic natural images, based on a natural language description along with an ROI mask. We achieve our goal by leveraging and combining a pretrained language-image model (CLIP), to steer the edit towards a user-provided text prompt, with a denoising diffusion probabilistic model (DDPM) to generate natural-looking results. To seamlessly fuse the edited region with the unchanged parts of the image, we spatially blend noised versions of the input image with the local text-guided diffusion latent at a progression of noise levels. In addition, we show that adding augmentations to the diffusion process mitigates adversarial results. We compare against several baselines and related methods, both qualitatively and quantitatively, and show that our method outperforms these solutions in terms of overall realism, ability to preserve the background and matching the text. Finally, we show several text-driven editing applications, including adding a new object to an image, removing/replacing/altering existing objects, background replacement, and image extrapolation. Code is available at: this https URL
* [[UniTune](https://arxiv.org/abs/2210.09477)]
    [[pdf](https://arxiv.org/pdf/2210.09477.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2210.09477/)]
    * Title: UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image
    * Year: 17 Oct `2022`
    * Authors: Dani Valevski, Matan Kalman, Yossi Matias, Yaniv Leviathan
    * Abstract: We present UniTune, a simple and novel method for general text-driven image editing. UniTune gets as input an arbitrary image and a textual edit description, and carries out the edit while maintaining high semantic and visual fidelity to the input image. UniTune uses text, an intuitive interface for art-direction, and does not require additional inputs, like masks or sketches. At the core of our method is the observation that with the right choice of parameters, we can fine-tune a large text-to-image diffusion model on a single image, encouraging the model to maintain fidelity to the input image while still allowing expressive manipulations. We used Imagen as our text-to-image model, but we expect UniTune to work with other large-scale models as well. We test our method in a range of different use cases, and demonstrate its wide applicability.
* [[DiffEdit](https://arxiv.org/abs/2210.11427)]
    [[pdf](https://arxiv.org/pdf/2210.11427.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2210.11427/)]
    * Title: DiffEdit: Diffusion-based semantic image editing with mask guidance
    * Year: 20 Oct `2022`
    * Authors: Guillaume Couairon, Jakob Verbeek, Holger Schwenk, Matthieu Cord
    * Abstract: Image generation has recently seen tremendous advances, with diffusion models allowing to synthesize convincing images for a large variety of text prompts. In this article, we propose DiffEdit, a method to take advantage of text-conditioned diffusion models for the task of semantic image editing, where the goal is to edit an image based on a text query. Semantic image editing is an extension of image generation, with the additional constraint that the generated image should be as similar as possible to a given input image. Current editing methods based on diffusion models usually require to provide a mask, making the task much easier by treating it as a conditional inpainting task. In contrast, our main contribution is able to automatically generate a mask highlighting regions of the input image that need to be edited, by contrasting predictions of a diffusion model conditioned on different text prompts. Moreover, we rely on latent inference to preserve content in those regions of interest and show excellent synergies with mask-based diffusion. DiffEdit achieves state-of-the-art editing performance on ImageNet. In addition, we evaluate semantic image editing in more challenging settings, using images from the COCO dataset as well as text-based generated images.
* [[SDEdit](https://arxiv.org/abs/2108.01073)]
    [[pdf](https://arxiv.org/pdf/2108.01073.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2108.01073/)]
    * Title: SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations
    * Year: 02 Aug `2021`
    * Authors: Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon
    * Abstract: Guided image synthesis enables everyday users to create and edit photo-realistic images with minimum effort. The key challenge is balancing faithfulness to the user input (e.g., hand-drawn colored strokes) and realism of the synthesized image. Existing GAN-based methods attempt to achieve such balance using either conditional GANs or GAN inversions, which are challenging and often require additional training data or loss functions for individual applications. To address these issues, we introduce a new image synthesis and editing method, Stochastic Differential Editing (SDEdit), based on a diffusion model generative prior, which synthesizes realistic images by iteratively denoising through a stochastic differential equation (SDE). Given an input image with user guide of any type, SDEdit first adds noise to the input, then subsequently denoises the resulting image through the SDE prior to increase its realism. SDEdit does not require task-specific training or inversions and can naturally achieve the balance between realism and faithfulness. SDEdit significantly outperforms state-of-the-art GAN-based methods by up to 98.09% on realism and 91.72% on overall satisfaction scores, according to a human perception study, on multiple tasks, including stroke-based image synthesis and editing as well as image compositing.
* [[RePaint](https://arxiv.org/abs/2201.09865)]
    [[pdf](https://arxiv.org/pdf/2201.09865.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2201.09865/)]
    * Title: RePaint: Inpainting using Denoising Diffusion Probabilistic Models
    * Year: 24 Jan `2022`
    * Authors: Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool
    * Abstract: Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask. Most existing approaches train for a certain distribution of masks, which limits their generalization capabilities to unseen mask types. Furthermore, training with pixel-wise and perceptual losses often leads to simple textural extensions towards the missing areas instead of semantically meaningful generation. In this work, we propose RePaint: A Denoising Diffusion Probabilistic Model (DDPM) based inpainting approach that is applicable to even extreme masks. We employ a pretrained unconditional DDPM as the generative prior. To condition the generation process, we only alter the reverse diffusion iterations by sampling the unmasked regions using the given image information. Since this technique does not modify or condition the original DDPM network itself, the model produces high-quality and diverse output images for any inpainting form. We validate our method for both faces and general-purpose image inpainting using standard and extreme masks. RePaint outperforms state-of-the-art Autoregressive, and GAN approaches for at least five out of six mask distributions. Github Repository: this http URL
* [[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)]
    [[pdf](https://arxiv.org/pdf/2207.12598.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2207.12598/)]
    * Title: Classifier-Free Diffusion Guidance
    * Year: 26 Jul `2022`
    * Authors: Jonathan Ho, Tim Salimans
    * Abstract: Classifier guidance is a recently introduced method to trade off mode coverage and sample fidelity in conditional diffusion models post training, in the same spirit as low temperature sampling or truncation in other types of generative models. Classifier guidance combines the score estimate of a diffusion model with the gradient of an image classifier and thereby requires training an image classifier separate from the diffusion model. It also raises the question of whether guidance can be performed without a classifier. We show that guidance can be indeed performed by a pure generative model without such a classifier: in what we call classifier-free guidance, we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates to attain a trade-off between sample quality and diversity similar to that obtained using classifier guidance.
* [[Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)]
    [[pdf](https://arxiv.org/pdf/2208.01626.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2208.01626/)]
    * Title: Prompt-to-Prompt Image Editing with Cross Attention Control
    * Year: 02 Aug `2022`
    * Authors: Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, Daniel Cohen-Or
    * Abstract: Recent large-scale text-driven synthesis models have attracted much attention thanks to their remarkable capabilities of generating highly diverse images that follow given text prompts. Such text-based synthesis methods are particularly appealing to humans who are used to verbally describe their intent. Therefore, it is only natural to extend the text-driven image synthesis to text-driven image editing. Editing is challenging for these generative models, since an innate property of an editing technique is to preserve most of the original image, while in the text-based models, even a small modification of the text prompt often leads to a completely different outcome. State-of-the-art methods mitigate this by requiring the users to provide a spatial mask to localize the edit, hence, ignoring the original structure and content within the masked region. In this paper, we pursue an intuitive prompt-to-prompt editing framework, where the edits are controlled by text only. To this end, we analyze a text-conditioned model in depth and observe that the cross-attention layers are the key to controlling the relation between the spatial layout of the image to each word in the prompt. With this observation, we present several applications which monitor the image synthesis by editing the textual prompt only. This includes localized editing by replacing a word, global editing by adding a specification, and even delicately controlling the extent to which a word is reflected in the image. We present our results over diverse images and prompts, demonstrating high-quality synthesis and fidelity to the edited prompts.

## Image Denoising

* [[DDPM](https://arxiv.org/abs/2006.11239)]
    [[pdf](https://arxiv.org/pdf/2006.11239.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2006.11239/)]
    * Title: Denoising Diffusion Probabilistic Models
    * Year: 19 Jun `2020`
    * Authors: Jonathan Ho, Ajay Jain, Pieter Abbeel
    * Abstract: We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at this https URL
* [[DDIM](https://arxiv.org/abs/2010.02502)]
    [[pdf](https://arxiv.org/pdf/2010.02502.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2010.02502/)]
    * Title: Denoising Diffusion Implicit Models
    * Year: 06 Oct `2020`
    * Authors: Jiaming Song, Chenlin Meng, Stefano Ermon
    * Abstract: Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a Markovian diffusion process. We construct a class of non-Markovian diffusion processes that lead to the same training objective, but whose reverse process can be much faster to sample from. We empirically demonstrate that DDIMs can produce high quality samples $10 \times$ to $50 \times$ faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, and can perform semantically meaningful image interpolation directly in the latent space.
* [[ILVR](https://arxiv.org/abs/2108.02938)]
    [[pdf](https://arxiv.org/pdf/2108.02938.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2108.02938/)]
    * Title: ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models
    * Year: 06 Aug `2021`
    * Authors: Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon
    * Abstract: Denoising diffusion probabilistic models (DDPM) have shown remarkable performance in unconditional image generation. However, due to the stochasticity of the generative process in DDPM, it is challenging to generate images with the desired semantics. In this work, we propose Iterative Latent Variable Refinement (ILVR), a method to guide the generative process in DDPM to generate high-quality images based on a given reference image. Here, the refinement of the generative process in DDPM enables a single DDPM to sample images from various sets directed by the reference image.

## Unclassified

* [[How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288)]
    [[pdf](https://arxiv.org/pdf/2101.03288.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2101.03288/)]
    * Title: How to Train Your Energy-Based Models
    * Year: 09 Jan `2021`
    * Authors: Yang Song, Diederik P. Kingma
    * Abstract: Energy-Based Models (EBMs), also known as non-normalized probabilistic models, specify probability density or mass functions up to an unknown normalizing constant. Unlike most other probabilistic models, EBMs do not place a restriction on the tractability of the normalizing constant, thus are more flexible to parameterize and can model a more expressive family of probability distributions. However, the unknown normalizing constant of EBMs makes training particularly difficult. Our goal is to provide a friendly introduction to modern approaches for EBM training. We start by explaining maximum likelihood training with Markov chain Monte Carlo (MCMC), and proceed to elaborate on MCMC-free approaches, including Score Matching (SM) and Noise Constrastive Estimation (NCE). We highlight theoretical connections among these three approaches, and end with a brief survey on alternative training methods, which are still under active research. Our tutorial is targeted at an audience with basic understanding of generative models who want to apply EBMs or start a research project in this direction.
* [[DiffusionCLIP](https://arxiv.org/abs/2110.02711)]
    [[pdf](https://arxiv.org/pdf/2110.02711.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2110.02711/)]
    * Title: DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation
    * Year: 06 Oct `2021`
    * Authors: Gwanghyun Kim, Taesung Kwon, Jong Chul Ye
    * Abstract: Recently, GAN inversion methods combined with Contrastive Language-Image Pretraining (CLIP) enables zero-shot image manipulation guided by text prompts. However, their applications to diverse real images are still difficult due to the limited GAN inversion capability. Specifically, these approaches often have difficulties in reconstructing images with novel poses, views, and highly variable contents compared to the training data, altering object identity, or producing unwanted image artifacts. To mitigate these problems and enable faithful manipulation of real images, we propose a novel method, dubbed DiffusionCLIP, that performs text-driven image manipulation using diffusion models. Based on full inversion capability and high-quality image generation power of recent diffusion models, our method performs zero-shot image manipulation successfully even between unseen domains and takes another step towards general application by manipulating images from a widely varying ImageNet dataset. Furthermore, we propose a novel noise combination method that allows straightforward multi-attribute manipulation. Extensive experiments and human evaluation confirmed robust and superior manipulation performance of our methods compared to the existing baselines. Code is available at this https URL.

### Mon Jan 23, 2023 Readings

* [[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)]
    [[pdf](https://arxiv.org/pdf/1503.03585.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1503.03585/)]
    * Title: Deep Unsupervised Learning using Nonequilibrium Thermodynamics
    * Year: 12 Mar `2015`
    * Authors: Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli
    * Abstract: A central problem in machine learning involves modeling complex data-sets using highly flexible families of probability distributions in which learning, sampling, inference, and evaluation are still analytically or computationally tractable. Here, we develop an approach that simultaneously achieves both flexibility and tractability. The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data. This approach allows us to rapidly learn, sample from, and evaluate probabilities in deep generative models with thousands of layers or time steps, as well as to compute conditional and posterior probabilities under the learned model. We additionally release an open source reference implementation of the algorithm.
* [[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)]
    [[pdf](https://arxiv.org/pdf/2208.11970.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2208.11970/)]
    * Title: Understanding Diffusion Models: A Unified Perspective
    * Year: 25 Aug `2022`
    * Authors: Calvin Luo
    * Abstract: Diffusion models have shown incredible capabilities as generative models; indeed, they power the current state-of-the-art models on text-conditioned image generation such as Imagen and DALL-E 2. In this work we review, demystify, and unify the understanding of diffusion models across both variational and score-based perspectives. We first derive Variational Diffusion Models (VDM) as a special case of a Markovian Hierarchical Variational Autoencoder, where three key assumptions enable tractable computation and scalable optimization of the ELBO. We then prove that optimizing a VDM boils down to learning a neural network to predict one of three potential objectives: the original source input from any arbitrary noisification of it, the original source noise from any arbitrarily noisified input, or the score function of a noisified input at any arbitrary noise level. We then dive deeper into what it means to learn the score function, and connect the variational perspective of a diffusion model explicitly with the Score-based Generative Modeling perspective through Tweedie's Formula. Lastly, we cover how to learn a conditional distribution using diffusion models via guidance.
* Denoising Diffusion Probabilistic Models
* High-Resolution Image Synthesis with Latent Diffusion Models
* Diffusion Models: A Comprehensive Survey of Methods and Applications

### Mon Feb 06, 2023 Readings

* [[An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)]
    [[pdf](https://arxiv.org/pdf/2208.01618.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2208.01618/)]
    * Title: An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion
    * Year: 02 Aug `2022`
    * Authors: Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, Daniel Cohen-Or
    * Abstract: Text-to-image models offer unprecedented freedom to guide creation through natural language. Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes. In other words, we ask: how can we use language-guided models to turn our cat into a painting, or imagine a new product based on our favorite toy? Here we present a simple approach that allows such creative freedom. Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model. These "words" can be composed into natural language sentences, guiding personalized creation in an intuitive way. Notably, we find evidence that a single word embedding is sufficient for capturing unique and varied concepts. We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks. Our code, data and new words will be available at: this https URL
