#! https://zhuanlan.zhihu.com/p/558911556
# [Notes][Vision][Segmentation] DeepMask

* url: https://arxiv.org/abs/1506.06204
* Title: Learning to Segment Object Candidates
* Year: 20 Jun `2015`
* Authors: Pedro O. Pinheiro, Ronan Collobert, Piotr Dollar
* Institutions: [Facebook AI Research]
* Abstract: Recent object detection systems rely on two critical steps: (1) a set of object proposals is predicted as efficiently as possible, and (2) this set of candidate proposals is then passed to an object classifier. Such approaches have been shown they can be fast, while achieving the state of the art in detection performance. In this paper, we propose a new way to generate object proposals, introducing an approach based on a discriminative convolutional network. Our model is trained jointly with two objectives: given an image patch, the first part of the system outputs a class-agnostic segmentation mask, while the second part of the system outputs the likelihood of the patch being centered on a full object. At test time, the model is efficiently applied on the whole test image and generates a set of segmentation masks, each of them being assigned with a corresponding object likelihood score. We show that our model yields significant improvements over state-of-the-art object proposal algorithms. In particular, compared to previous approaches, our model obtains substantially higher object recall using fewer proposals. We also show that our model is able to generalize to unseen categories it has not seen during training. Unlike all previous approaches for generating object masks, we do not rely on edges, superpixels, or any other form of low-level segmentation.

----------------------------------------------------------------------------------------------------

## 1 Introduction

> Object proposal algorithms aim to find diverse regions in an image which are likely to contain objects. For efficiency and detection performance reasons, an ideal proposal method should possess three key characteristics:
> 1. high recall (i.e., the proposed regions should contain the maximum number of possible objects),
> 2. the high recall should be achieved with the minimum number of regions possible, and
> 3. the proposed regions should match the objects as accurately as possible.

> In this paper, we present an object proposal algorithm based on Convolutional Networks (ConvNets) [20] that satisfies these constraints better than existing approaches.

> Given an input image patch, our algorithm generates a class-agnostic mask and an associated score which estimates the likelihood of the patch fully containing a centered object (without any notion of an object category). The core of our model is a ConvNet which jointly predicts the mask and the object score. A large part of the network is shared between those two tasks: only the last few network layers are specialized for separately outputting a mask and score prediction. The model is trained by optimizing a cost function that targets both tasks simultaneously.

> Unlike all previous approaches for generating segmentation proposals, we do not rely on edges, superpixels, or any other form of low-level segmentation. Our approach is the first to learn to generate segmentation proposals directly from raw image data.

## 2 Related Work

> Although our model shares high level similarities with these approaches (we generate a set of ranked segmentation proposals), these results are achieved quite differently. All previous approaches for generating segmentation masks, including [17] which has a learning component, rely on low-level segmentations such as superpixels or edges. Instead, we propose a data-driven discriminative approach based on a deep-network architecture to obtain our segmentation proposals.

## 3 DeepMask Proposals

> Our object proposal method predicts a segmentation mask given an input patch, and assigns a score corresponding to how likely the patch is to contain an object.

> Both mask and score predictions are achieved with a single convolutional network.

> During training, the two tasks are learned jointly.

Notations:
* Let $x^{(k)} \in \mathbb{R}^{H \times W \times C}$ denote the $k$-th RGB input patch in the training set.
* Let $y^{(k)} \in \{\pm1\}$ denote the $k$-th label in the training set, corresponding to $x^{(k)}$.
* Let $m^{(k)} \in \{\pm1\}^{H \times W}$ denote the $k$-th binary mask in the training set, corresponding to $x^{(k)}$.

How are the $y^{(k)}$'s defined?

> A patch $x^{(k)}$ is given label $y^{(k)} = 1$ if it satisfies the following constraints:
> 1. the patch contains an object roughly centered in the input patch
> 2. the object is fully contained in the patch and in a given scale range

> Otherwise, $y^{(k)} = −1$, even if an object is partially present. The positional and scale tolerance used in our experiments are given shortly.

How are the $m^{(k)}$'s defined?

> Assuming $y^{(k)} = 1$, the ground truth mask $m^{(k)}$ has positive values only for the pixels that are part of the single object located in the center of the patch.

> If $y^{(k)} = -1$ the mask is not used.

### 3.1 Network Architecture

> The parameters for the layers shared between the mask prediction and the object score prediction are initialized with a network that was pre-trained to perform classification on the ImageNet dataset [5]. This model is then fine-tuned for generating object proposals during training.

> We choose the VGGA architecture [27] which consists of eight 3 x 3 convolutional layers (followed by ReLU nonlinearities) and five 2 x 2 max-pooling layers and has shown excellent performance.

> As we are interested in inferring segmentation masks, the spatial information provided in the convolutional feature maps is important. We therefore remove all the final fully connected layers of the VGG-A model. Additionally we also discard the last max-pooling layer.

**Segmentation**

> The branch of the network dedicated to segmentation is composed of a single 1 x 1 convolution layer (and ReLU non-linearity) followed by a classification layer.

> The classification layer consists of $h \times w$ pixel classifiers, each responsible for indicating whether a given pixel belongs to the object in the center of the patch. Note that each pixel classifier in the output plane must be able to utilize information contained in the entire feature map, and thus have a complete view of the object. This is critical because unlike in semantic segmentation, our network must output a mask for a single object even when multiple objects are present.

> For the classification layer one could use either locally or fully connected pixel classifiers. Both options have drawbacks: in the former each classifier has only a partial view of the object while in the latter the classifiers have a massive number of redundant parameters.

> Instead, we opt to decompose the classification layer into two linear layers with no non-linearity in between.

> Finally, to further reduce model capacity, we set the output of the classification layer to be $h^{o} \times w^{o}$ with $h^{o} < h$ and $w_{o} < w$ and upsample the output to $h \times w$ to match the input dimensions.

**Scoring**

> The second branch of the network is dedicated to predicting if an image patch satisfies constraints (i) and (ii): that is if an object is centered in the patch and at the appropriate scale.

> It is composed of a 2 x 2 max-pooling layer, followed by two fully connected (plus ReLU non-linearity) layers. The final output is a single 'objectness' score indicating the presence of an object in the center of the input patch (and at the appropriate scale).

### 3.2 Joint Learning

Notations:
* Let $f^{\text{seg}}$ denote the segmentation branch of the network.
* Let $f^{\text{scr}}$ denote the scoring branch of the network.
* Let $\theta$ denote the parameters of the network.

Segmentation loss:
$$\mathcal{L}_{\text{seg}}(x^{(k)}, m^{(k)}, y^{(k)}; \theta) := \frac{1+y_{k}}{2w^{o}h^{o}}\sum_{ij}\log\bigg[1+\exp(-m^{(k)}_{ij}f^{\text{seg}}_{ij}(x^{(k)}))\bigg].$$

Scoring loss:
$$\mathcal{L}_{\text{scr}}(x^{(k)}, m^{(k)}, y^{(k)}; \theta) := \log\bigg[1+\exp(-y_{k}f^{\text{scr}}(x^{(k)}))\bigg].$$

Total loss:

> The loss function is a sum of binary logistic regression losses, one for each location of the segmentation network and one for the object score, over all training triplets $(x^{(k)}, m^{(k)}, y^{(k)})$.

$$\mathcal{L}(\theta) := \sum_{k}\bigg[\mathcal{L}_{\text{seg}}(x^{(k)}, m^{(k)}, y^{(k)}, \theta) + \lambda\mathcal{L}_{\text{scr}}(x^{(k)}, m^{(k)}, y^{(k)}, \theta)\bigg].$$

### 3.3 Full Scene Inference

> During full image inference, we apply the model densely at multiple locations and scales. This is necessary so that for each object in the image we test at least one patch that fully contains the object (roughly centered and at the appropriate scale), satisfying the two assumptions made during training. This procedure gives a segmentation mask and object score at each image location.

> Given an input test image of size $h^{t} \times w^{t}$, the segmentation and object network generate outputs of dimension $\frac{h^{t}}{16} \times \frac{w^{t}}{16}$ and $\frac{h^{t}}{32} \times \frac{w^{t}}{32}$, respectively. In order to achieve a one-to-one mapping between the mask prediction and object score, we apply the interleaving trick right before the last max-pooling layer for the scoring branch to double its output resolution (we use exactly the implementation described in [26]).

### 3.4 Implementation Details

## 4 Experimental Results

## 5 Conclusion

> In this paper, we propose an innovative framework to generate segmentation object proposals directly from image pixels. At test time, the model is applied densely over the entire image at multiple scales and generates a set of ranked segmentation proposals.

----------------------------------------------------------------------------------------------------

## References

* O Pinheiro, Pedro O., Ronan Collobert, and Piotr Dollár. "Learning to segment object candidates." *Advances in neural information processing systems* 28 (2015).

## Further Reading

* [3] DeepLabv1
* [6] MultiBox
* [9] Fast R-CNN
* [10] R-CNN
* [11] Hypercolumns
* [15] Spatial Pyramid Pooling (SPP)
* [18] [AlexNet](https://zhuanlan.zhihu.com/p/565285454)
* [25] Faster R-CNN
* [26] OverFeat
* [27] [VGGNet](https://zhuanlan.zhihu.com/p/563314926)
* [28] Dropout
* [29] [InceptionNetV1/GoogLeNet](https://zhuanlan.zhihu.com/p/564141144)
* [30] MSC-MultiBox
