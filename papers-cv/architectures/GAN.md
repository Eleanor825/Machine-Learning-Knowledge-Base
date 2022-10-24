# [Papers][Vision] GAN <!-- omit in toc -->

count=6

## Table of Contents <!-- omit in toc -->

- [Basics](#basics)
- [Unclassified](#unclassified)
- [Progressive Learning (EfficientNetV2, 2021)](#progressive-learning-efficientnetv2-2021)
- [Adversarial Learning (EfficientNetV2, 2021)](#adversarial-learning-efficientnetv2-2021)

----------------------------------------------------------------------------------------------------

## Basics

* [[GAN](https://arxiv.org/abs/1406.2661)]
    [[pdf](https://arxiv.org/pdf/1406.2661.pdf)]
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
* [[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)]
    [[pdf](https://arxiv.org/pdf/1703.10593.pdf)]
    * Title: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
    * Year: 30 Mar `2017`
    * Authors: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
    * Abstract: Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.
* [[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)]
    [[pdf](https://arxiv.org/pdf/1611.07004.pdf)]
    * Title: Image-to-Image Translation with Conditional Adversarial Networks
    * Year: 21 Nov `2016`
    * Authors: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
    * Abstract: We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.
* [[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)]
    [[pdf](https://arxiv.org/pdf/1704.00028.pdf)]
    * Title: Improved Training of Wasserstein GANs
    * Year: 31 Mar `2017`
    * Authors: Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
    * Abstract: Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.
* [[Wasserstein GAN](https://arxiv.org/abs/1701.07875)]
    [[pdf](https://arxiv.org/pdf/1701.07875.pdf)]
    * Title: Wasserstein GAN
    * Year: 26 Jan `2017`
    * Authors: Martin Arjovsky, Soumith Chintala, Léon Bottou
    * Abstract: We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.
* [[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)]
    [[pdf](https://arxiv.org/pdf/1511.06434.pdf)]
    * Title: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    * Year: 19 Nov `2015`
    * Authors: Alec Radford, Luke Metz, Soumith Chintala
    * Abstract: In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.
* [[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.03242)]
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

## Unclassified

* [[Style GAN](https://arxiv.org/abs/1812.04948)]
    [[pdf](https://arxiv.org/pdf/1812.04948.pdf)]
    * Title: A Style-Based Generator Architecture for Generative Adversarial Networks
    * Year: 12 Dec `2018`
    * Authors: Tero Karras, Samuli Laine, Timo Aila
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
* [[APE-GAN: Adversarial Perturbation Elimination with GAN](https://arxiv.org/abs/1707.05474)]
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
* [[AdaGAN: Boosting Generative Models](https://arxiv.org/abs/1701.02386)]
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

## Progressive Learning (EfficientNetV2, 2021)

* [[Progressive Growing of GAN](https://arxiv.org/abs/1710.10196)]
    [[pdf](https://arxiv.org/pdf/1710.10196.pdf)]
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
