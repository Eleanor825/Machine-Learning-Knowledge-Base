#! https://zhuanlan.zhihu.com/p/563314926
# [Notes][Vision][CNN] VGG

* url: https://arxiv.org/abs/1409.1556
* Title: Very Deep Convolutional Networks for Large-Scale Image Recognition
* Year: 04 Sep `2014`
* Authors: Karen Simonyan, Andrew Zisserman
* Institutions: [Visual Geometry Group, Department of Engineering Science, University of Oxford]
* Abstract: In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

----------------------------------------------------------------------------------------------------

## 1 INTRODUCTION

> In this paper, we address another important aspect of ConvNet architecture design – its depth.

> To this end, we fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small (3x3) convolution filters in all layers.

## 2 CONVNET CONFIGURATIONS

### 2.1 ARCHITECTURE

> The only pre-processing we do is subtracting the mean RGB value, computed on the training set, from each pixel.

> The image is passed through a stack of convolutional (conv.) layers, where we use filters with a very small receptive field: 3x3 (which is the smallest size to capture the notion of left/right, up/down, center).

> Max-pooling is performed over a 2x2 pixel window, with stride 2.

> All hidden layers are equipped with the rectification (ReLU (Krizhevsky et al., 2012)) non-linearity.

### 2.2 CONFIGURATIONS

> The width of conv. layers (the number of channels) is rather small, starting from 64 in the first layer and then increasing by a factor of 2 after each max-pooling layer, until it reaches 512.

### 2.3 DISCUSSION

Smaller Kernel Size

> Rather than using relatively large receptive fields in the first conv. layers (e.g. 11x11 with stride 4 in (Krizhevsky et al., 2012), or 7x7 with stride 2 in (Zeiler & Fergus, 2013; Sermanet et al., 2014)), we use very small 3x3 receptive fields throughout the whole net, which are convolved with the input at every pixel (with stride 1).

> It is easy to see that a stack of two 3x3 conv. layers (without spatial pooling in between) has an effective receptive field of 5x5; three such layers have a 7x7 effective receptive field.

> So what have we gained by using, for instance, a stack of three 3x3 conv. layers instead of a single 7x7 layer?
> 1. First, we incorporate three non-linear rectification layers instead of a single one, which makes the decision function more discriminative.
> 2. Second, we decrease the number of parameters.

> This can be seen as imposing a regularisation on the 7x7 conv. filters, forcing them to have a decomposition through the 3x3 filters (with non-linearity injected in between).

1x1 Convolutions

> The incorporation of 1x1 conv. layers (configuration C, Table 1) is a way to increase the non-linearity of the decision function without affecting the receptive fields of the conv. layers. Even though in our case the 1x1 convolution is essentially a linear projection onto the space of the same dimensionality (the number of input and output channels is the same), an additional non-linearity is introduced by the rectification function.

## 3 CLASSIFICATION FRAMEWORK

### 3.1 TRAINING

> We conjecture that in spite of the larger number of parameters and the greater depth of our nets compared to (Krizhevsky et al., 2012), the nets required less epochs to converge due to
> 1. implicit regularisation imposed by greater depth and smaller conv. filter sizes;
> 2. pre-initialisation of certain layers.

Network Initialization

> The initialisation of the network weights is important, since bad initialisation can stall learning due to the instability of gradient in deep nets. To circumvent this problem, we began with training the configuration A (Table 1), shallow enough to be trained with random initialisation. Then, when training deeper architectures, we initialised the first four convolutional layers and the last three fully-connected layers with the layers of net A (the intermediate layers were initialised randomly).

Cropping

> To obtain the fixed-size 224x224 ConvNet input images, they were randomly cropped from rescaled training images (one crop per image per SGD iteration).

**Training image size**

> Let $S$ be the smallest side of an `isotropically-rescaled` training image, from which the ConvNet input is cropped (we also refer to $S$ as the `training scale`). While the crop size is fixed to 224x224, in principle $S$ can take on any value not less than 224.

> We consider two approaches for setting the training scale $S$.
> 1. The first is to fix $S$, which corresponds to single-scale training (note that image content within the sampled crops can still represent multi-scale image statistics).
> 2. The second approach to setting $S$ is multi-scale training, where each training image is individually rescaled by randomly sampling $S$ from a certain range $[S_{min},S_{max}]$ (we used $S_{min}=256$ and $S_{max}=512$). Since objects in images can be of different size, it is beneficial to take this into account during training.

> This can also be seen as `training set augmentation` by `scale jittering`, where a single model is trained to recognise objects over a wide range of scales.

### 3.2 TESTING

> At test time, given a trained ConvNet and an input image, it is classified in the following way.
> 1. First, it is `isotropically rescaled` to a pre-defined smallest image side, denoted as $Q$ (we also refer to it as the `test scale`).
> 2. Then, the network is applied densely over the rescaled test image in a way similar to (Sermanet et al., 2014). Namely, the fully-connected layers are first converted to convolutional layers (the first FC layer to a 7x7 conv. layer, the last two FC layers to 1x1 conv. layers). The resulting fully-convolutional net is then applied to the whole (uncropped) image. The result is a class score map with the number of channels equal to the number of classes, and a variable spatial resolution, dependent on the input image size.
> 3. Finally, to obtain a fixed-size vector of class scores for the image, the class score map is spatially averaged (sum-pooled).

> Since the fully-convolutional network is applied over the whole image, there is no need to sample multiple crops at test time (Krizhevsky et al., 2012), which is less efficient as it requires network re-computation for each crop.

### 3.3 IMPLEMENTATION DETAILS

## 4 CLASSIFICATION EXPERIMENTS

### 4.1 SINGLE SCALE EVALUATION

### 4.2 MULTI-SCALE EVALUATION

### 4.3 MULTI-CROP EVALUATION

> In Table 5 we compare dense ConvNet evaluation with mult-crop evaluation (see Sect. 3.2 for details).

> the two approaches are indeed complementary, as their combination outperforms each of them.

### 4.4 CONVNET FUSION

### 4.5 COMPARISON WITH THE STATE OF THE ART

## 5 CONCLUSION

----------------------------------------------------------------------------------------------------

## References

* Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." *arXiv preprint arXiv:1409.1556* (2014).

## Further Reading

* [Krizhevsky et al., 2012] AlexNet
* [Zeiler & Fergus, 2013] ZFNet
* [Sermanet et al., 2014] OverFeat
* [Szegedy et al., 2014] Inception-v1/GoogLeNet
