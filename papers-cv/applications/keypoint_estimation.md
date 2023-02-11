<span style="font-family:monospace">

# Papers in Computer Vision - Keypoint Estimation

count: 2

* [Part-based R-CNNs for Fine-grained Category Detection](https://arxiv.org/abs/1407.3867)
    * Title: Part-based R-CNNs for Fine-grained Category Detection
    * Year: 15 Jul 2014
    * Author: Ning Zhang
    * Abstract: Semantic part localization can facilitate fine-grained categorization by explicitly isolating subtle appearance differences associated with specific object parts. Methods for pose-normalized representations have been proposed, but generally presume bounding box annotations at test time due to the difficulty of object detection. We propose a model for fine-grained categorization that overcomes these limitations by leveraging deep convolutional features computed on bottom-up region proposals. Our method learns whole-object and part detectors, enforces learned geometric constraints between them, and predicts a fine-grained category from a pose-normalized representation. Experiments on the Caltech-UCSD bird dataset confirm that our method outperforms state-of-the-art fine-grained categorization methods in an end-to-end evaluation without requiring a bounding box at test time.
* [Do Convnets Learn Correspondence?](https://arxiv.org/abs/1411.1091)
    * Title: Do Convnets Learn Correspondence?
    * Year: 04 Nov 2014
    * Author: Jonathan Long
    * Abstract: Convolutional neural nets (convnets) trained from massive labeled datasets have substantially improved the state-of-the-art in image classification and object detection. However, visual understanding requires establishing correspondence on a finer level than object category. Given their large pooling regions and training from whole-image labels, it is not clear that convnets derive their success from an accurate correspondence model which could be used for precise localization. In this paper, we study the effectiveness of convnet activation features for tasks requiring correspondence. We present evidence that convnet features localize at a much finer scale than their receptive field sizes, that they can be used to perform intraclass alignment as well as conventional hand-engineered features, and that they outperform conventional features in keypoint prediction on objects from PASCAL VOC 2011.
