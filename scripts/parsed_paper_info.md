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
