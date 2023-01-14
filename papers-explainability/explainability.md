# [Papers] Explainable AI <!-- omit in toc -->

count=5

## Table of Contents <!-- omit in toc -->
- [From the VoG Paper](#from-the-vog-paper)

----------------------------------------------------------------------------------------------------

## From the VoG Paper

* [[A Benchmark for Interpretability Methods in Deep Neural Networks](https://arxiv.org/abs/1806.10758)]
    [[pdf](https://arxiv.org/pdf/1806.10758.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1806.10758/)]
    * Title: A Benchmark for Interpretability Methods in Deep Neural Networks
    * Year: 28 Jun `2018`
    * Authors: Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, Been Kim
    * Abstract: We propose an empirical measure of the approximate accuracy of feature importance estimates in deep neural networks. Our results across several large-scale image classification datasets show that many popular interpretability methods produce estimates of feature importance that are not better than a random designation of feature importance. Only certain ensemble based approaches---VarGrad and SmoothGrad-Squared---outperform such a random assignment of importance. The manner of ensembling remains critical, we show that some approaches do no better then the underlying method but carry a far higher computational burden.
* [[Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)]
    [[pdf](https://arxiv.org/pdf/1704.02685.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1704.02685/)]
    * Title: Learning Important Features Through Propagating Activation Differences
    * Year: 10 Apr `2017`
    * Authors: Avanti Shrikumar, Peyton Greenside, Anshul Kundaje
    * Abstract: The purported "black box" nature of neural networks is a barrier to adoption in applications where interpretability is essential. Here we present DeepLIFT (Deep Learning Important FeaTures), a method for decomposing the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons in the network to every feature of the input. DeepLIFT compares the activation of each neuron to its 'reference activation' and assigns contribution scores according to the difference. By optionally giving separate consideration to positive and negative contributions, DeepLIFT can also reveal dependencies which are missed by other approaches. Scores can be computed efficiently in a single backward pass. We apply DeepLIFT to models trained on MNIST and simulated genomic data, and show significant advantages over gradient-based methods. Video tutorial: this http URL, ICML slides: this http URL, ICML talk: this https URL, code: this http URL.
* [[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)]
    [[pdf](https://arxiv.org/pdf/1312.6034.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1312.6034/)]
    * Title: Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
    * Year: 20 Dec `2013`
    * Authors: Karen Simonyan, Andrea Vedaldi, Andrew Zisserman
    * Abstract: This paper addresses the visualisation of image classification models, learnt using deep Convolutional Networks (ConvNets). We consider two visualisation techniques, based on computing the gradient of the class score with respect to the input image. The first one generates an image, which maximises the class score [Erhan et al., 2009], thus visualising the notion of the class, captured by a ConvNet. The second technique computes a class saliency map, specific to a given image and class. We show that such maps can be employed for weakly supervised object segmentation using classification ConvNets. Finally, we establish the connection between the gradient-based ConvNet visualisation methods and deconvolutional networks [Zeiler et al., 2013].
* [[SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)]
    [[pdf](https://arxiv.org/pdf/1706.03825.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1706.03825/)]
    * Title: SmoothGrad: removing noise by adding noise
    * Year: 12 Jun `2017`
    * Authors: Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg
    * Abstract: Explaining the output of a deep network remains a challenge. In the case of an image classifier, one type of explanation is to identify pixels that strongly influence the final decision. A starting point for this strategy is the gradient of the class score function with respect to the input image. This gradient can be interpreted as a sensitivity map, and there are several techniques that elaborate on this basic idea. This paper makes two contributions: it introduces SmoothGrad, a simple method that can help visually sharpen gradient-based sensitivity maps, and it discusses lessons in the visualization of these maps. We publish the code for our experiments and a website with our results.
* [[Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)]
    [[pdf](https://arxiv.org/pdf/1703.01365.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.01365/)]
    * Title: Axiomatic Attribution for Deep Networks
    * Year: 04 Mar `2017`
    * Authors: Mukund Sundararajan, Ankur Taly, Qiqi Yan
    * Abstract: We study the problem of attributing the prediction of a deep network to its input features, a problem previously studied by several other works. We identify two fundamental axioms---Sensitivity and Implementation Invariance that attribution methods ought to satisfy. We show that they are not satisfied by most known attribution methods, which we consider to be a fundamental weakness of those methods. We use the axioms to guide the design of a new attribution method called Integrated Gradients. Our method requires no modification to the original network and is extremely simple to implement; it just needs a few calls to the standard gradient operator. We apply this method to a couple of image models, a couple of text models and a chemistry model, demonstrating its ability to debug networks, to extract rules from a network, and to enable users to engage with models better.
