# [Papers][Vision] Other Tasks

## Image Super-Resolution

* [Learning a Deep Convolutional Network for Image Super-Resolution](https://link.springer.com/chapter/10.1007/978-3-319-10593-2_13)
    * Title: Learning a Deep Convolutional Network for Image Super-Resolution

## Depth Map Prediction

* [[Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields](https://arxiv.org/abs/1502.07411)]
    [[pdf](https://arxiv.org/pdf/1502.07411.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1502.07411/)]
    * Title: Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields
    * Year: 26 Feb `2015`
    * Authors: Fayao Liu, Chunhua Shen, Guosheng Lin, Ian Reid
    * Abstract: In this article, we tackle the problem of depth estimation from single monocular images. Compared with depth estimation using multiple images such as stereo depth perception, depth from monocular images is much more challenging. Prior work typically focuses on exploiting geometric priors or additional sources of information, most using hand-crafted features. Recently, there is mounting evidence that features from deep convolutional neural networks (CNN) set new records for various vision applications. On the other hand, considering the continuous characteristic of the depth values, depth estimations can be naturally formulated as a continuous conditional random field (CRF) learning problem. Therefore, here we present a deep convolutional neural field model for estimating depths from single monocular images, aiming to jointly explore the capacity of deep CNN and continuous CRF. In particular, we propose a deep structured learning scheme which learns the unary and pairwise potentials of continuous CRF in a unified deep CNN framework. We then further propose an equally effective model based on fully convolutional networks and a novel superpixel pooling method, which is $\sim 10$ times faster, to speedup the patch-wise convolutions in the deep model. With this more efficient model, we are able to design deeper networks to pursue better performance. Experiments on both indoor and outdoor scene datasets demonstrate that the proposed method outperforms state-of-the-art depth estimation approaches.
    * Comments:
        * > (2016, RefineNet) The depth estimation method [34] employs super-pixel pooling to output high-resolution prediction.
* [[Deep Convolutional Neural Fields for Depth Estimation from a Single Image](https://arxiv.org/abs/1411.6387)]
    [[pdf](https://arxiv.org/pdf/1411.6387.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1411.6387/)]
    * Title: Deep Convolutional Neural Fields for Depth Estimation from a Single Image
    * Year: 24 Nov `2014`
    * Authors: Fayao Liu, Chunhua Shen, Guosheng Lin
    * Abstract: We consider the problem of depth estimation from a single monocular image in this work. It is a challenging task as no reliable depth cues are available, e.g., stereo correspondences, motions, etc. Previous efforts have been focusing on exploiting geometric priors or additional sources of information, with all using hand-crafted features. Recently, there is mounting evidence that features from deep convolutional neural networks (CNN) are setting new records for various vision applications. On the other hand, considering the continuous characteristic of the depth values, depth estimations can be naturally formulated into a continuous conditional random field (CRF) learning problem. Therefore, we in this paper present a deep convolutional neural field model for estimating depths from a single image, aiming to jointly explore the capacity of deep CNN and continuous CRF. Specifically, we propose a deep structured learning scheme which learns the unary and pairwise potentials of continuous CRF in a unified deep CNN framework. The proposed method can be used for depth estimations of general scenes with no geometric priors nor any extra information injected. In our case, the integral of the partition function can be analytically calculated, thus we can exactly solve the log-likelihood optimization. Moreover, solving the MAP problem for predicting depths of a new image is highly efficient as closed-form solutions exist. We experimentally demonstrate that the proposed method outperforms state-of-the-art depth estimation methods on both indoor and outdoor scene datasets.
* [[Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture](https://arxiv.org/abs/1411.4734)]
    [[pdf](https://arxiv.org/pdf/1411.4734.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1411.4734/)]
    * Title: Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture
    * Year: 18 Nov `2014`
    * Authors: David Eigen, Rob Fergus
    * Abstract: In this paper we address three different computer vision tasks using a single basic architecture: depth prediction, surface normal estimation, and semantic labeling. We use a multiscale convolutional network that is able to adapt easily to each task using only small modifications, regressing from the input image to the output map directly. Our method progressively refines predictions using a sequence of scales, and captures many image details without any superpixels or low-level segmentation. We achieve state-of-the-art performance on benchmarks for all three tasks.
* [[Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/abs/1406.2283)]
    [[pdf](https://arxiv.org/pdf/1406.2283.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1406.2283/)]
    * Title: Depth Map Prediction from a Single Image using a Multi-Scale Deep Network
    * Year: 09 Jun `2014`
    * Authors: David Eigen, Christian Puhrsch, Rob Fergus
    * Abstract: Predicting depth is an essential component in understanding the 3D geometry of a scene. While for stereo images local correspondence suffices for estimation, finding depth relations from a single image is less straightforward, requiring integration of both global and local information from various cues. Moreover, the task is inherently ambiguous, with a large source of uncertainty coming from the overall scale. In this paper, we present a new method that addresses this task by employing two deep network stacks: one that makes a coarse global prediction based on the entire image, and another that refines this prediction locally. We also apply a scale-invariant error to help measure depth relations rather than scale. By leveraging the raw datasets as large sources of training data, our method achieves state-of-the-art results on both NYU Depth and KITTI, and matches detailed depth boundaries without the need for superpixelation.
    * Comments:
        * > (2015, SegNet) The authors in [50] discuss the need for learning to upsample from low resolution feature maps which is the central topic of this paper.

## Optical Flow Estimation

* [FlowNet](https://arxiv.org/abs/1504.06852)
    [[pdf](https://arxiv.org/pdf/1504.06852.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1504.06852/)]
    * Title: FlowNet: Learning Optical Flow with Convolutional Networks
    * Year: 26 Apr `2015`
    * Authors: Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox
    * Institutions: [University of Freiburg], [Technical University of Munich]
    * Abstract: Convolutional neural networks (CNNs) have recently been very successful in a variety of computer vision tasks, especially on those linked to recognition. Optical flow estimation has not been among the tasks where CNNs were successful. In this paper we construct appropriate CNNs which are capable of solving the optical flow estimation problem as a supervised learning task. We propose and compare two architectures: a generic architecture and another one including a layer that correlates feature vectors at different image locations. Since existing ground truth data sets are not sufficiently large to train a CNN, we generate a synthetic Flying Chairs dataset. We show that networks trained on this unrealistic data still generalize very well to existing datasets such as Sintel and KITTI, achieving competitive accuracy at frame rates of 5 to 10 fps.
