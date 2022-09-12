# [Papers][Vision] MLP Architectures

count: 2

* [NIN](https://arxiv.org/abs/1312.4400)
    * Title: Network In Network
    * Year: 16 Dec `2013`
    * Authors: Min Lin, Qiang Chen, Shuicheng Yan
    * Abstract: We propose a novel deep network structure called "Network In Network" (NIN) to enhance model discriminability for local patches within the receptive field. The conventional convolutional layer uses linear filters followed by a nonlinear activation function to scan the input. Instead, we build micro neural networks with more complex structures to abstract the data within the receptive field. We instantiate the micro neural network with a multilayer perceptron, which is a potent function approximator. The feature maps are obtained by sliding the micro networks over the input in a similar manner as CNN; they are then fed into the next layer. Deep NIN can be implemented by stacking multiple of the above described structure. With enhanced local modeling via the micro network, we are able to utilize global average pooling over feature maps in the classification layer, which is easier to interpret and less prone to overfitting than traditional fully connected layers. We demonstrated the state-of-the-art classification performances with NIN on CIFAR-10 and CIFAR-100, and reasonable performances on SVHN and MNIST datasets.
    * Comments:
        * > (2014, Inception-v1/GoogLeNet) Network-in-Network is an approach proposed by Lin et al. [12] in order to increase the representational power of neural networks. When applied to convolutional layers, the method could be viewed as additional 1x1 convolutional layers followed typically by the rectified linear activation [9].
        * > (2016, Deep Roots) Lin et al. [19] proposed a method to reduce the dimensionality of convolutional feature maps. By using relatively cheap '1×1' convolutional layers (i.e. layers comprising $d$ filters of size $1 \times 1 \times c$, where $d < c$), they learn to map feature maps into lower dimensional spaces, i.e. to new feature maps with fewer channels. Subsequent spatial filters operating on this lower dimensional input space require significantly less computation.
* [MLP-Mixer](https://arxiv.org/abs/2105.01601)
    * Title: MLP-Mixer: An all-MLP Architecture for Vision
    * Year: 04 May `2021`
    * Author: Ilya Tolstikhin
    * Abstract: Convolutional Neural Networks (CNNs) are the go-to model for computer vision. Recently, attention-based networks, such as the Vision Transformer, have also become popular. In this paper we show that while convolutions and attention are both sufficient for good performance, neither of them are necessary. We present MLP-Mixer, an architecture based exclusively on multi-layer perceptrons (MLPs). MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information). When trained on large datasets, or with modern regularization schemes, MLP-Mixer attains competitive scores on image classification benchmarks, with pre-training and inference cost comparable to state-of-the-art models. We hope that these results spark further research beyond the realms of well established CNNs and Transformers.
