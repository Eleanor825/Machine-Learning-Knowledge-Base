# [Notes][Vision][Detection] YOLOv1

* url: https://arxiv.org/abs/1506.02640
* Title: You Only Look Once: Unified, Real-Time Object Detection
* Year: 08 Jun `2015`
* Authors: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
* Abstract: We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. Compared to state-of-the-art detection systems, YOLO makes more localization errors but is far less likely to predict false detections where nothing exists. Finally, YOLO learns very general representations of objects. It outperforms all other detection methods, including DPM and R-CNN, by a wide margin when generalizing from natural images to artwork on both the Picasso Dataset and the People-Art Dataset.

----------------------------------------------------------------------------------------------------

## 2 Unified Detection

> Our network uses features from the entire image to predict each bounding box. It also predicts all bounding boxes across all classes for an image simultaneously. This means our network reasons globally about the full image and all the objects in the image.

$$confidence := \Pr(Object) * IoU$$

confidence being 0 means there is no object.

confidence being not 0 means the IoU of the predicted bounding box and the ground truth box.

Notations:
* Let $S \in \mathbb{Z}_{++}$ denote the number of grids in a single row/col.
The whole image is divided into $S \times S$ grids.
* Let $B \in \mathbb{Z}_{++}$ denote the number of bounding box predictions per grid.
* Let $C \in \mathbb{Z}_{++}$ denote the number of classes.

* the output of the network is of shape $S \times S \times (B \times 5 + C)$.
* multiply the class probabilities with the confidence of the bounding boxes to get the class-specific confidence scores for each bounding box.

### 2.2 Training

Loss function:
* > We use sum-squared error because it is easy to optimize, however it does not perfectly align with our goal of maximizing average precision.
* > It weights localization error equally with classification error which may not be ideal. Also, in every image many grid cells do not contain any object. This pushes the "confidence" scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability, causing training to diverge early on.
* > To remedy this, we increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions for boxes that don’t contain objects.
* > Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes.
* > To partially address this we predict the square root of the bounding box width and height instead of the width and height directly.

Notations:
* Let
$$\begin{aligned}
    \mathcal{L}_{\text{box}}(\text{pred}, \text{true})
        & := \lambda_{\text{coord}}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathbb{1}_{i,j}^{\text{obj}}
        \bigg[(x_{i}-\hat{x}_{i})^{2} + (y_{i}-\hat{y}_{i})^{2}\bigg] \\
        & + \lambda_{\text{coord}}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathbb{1}_{i,j}^{\text{obj}}
        \bigg[(\sqrt{w_{i}}-\sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}}-\sqrt{\hat{h}_{i}})^{2}\bigg], \\
    \mathcal{L}_{\text{conf}}(\text{pred}, \text{true})
        & := \sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathbb{1}_{i,j}^{\text{obj}}(C_{i}-\hat{C}_{i})^{2}
        + \lambda_{\text{noobj}}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathbb{1}_{i,j}^{\text{noobj}}(C_{i}-\hat{C}_{i})^{2}, \\
    \mathcal{L}_{\text{cls}}(\text{pred}, \text{true})
        & := \sum_{i=0}^{S^{2}}\sum_{c\in\text{classes}}\mathbb{1}_{i}^{\text{obj}}(p_{i}(c)-\hat{p}_{i}(c))^{2}, \\
    \mathcal{L}(\text{pred}, \text{true})
        & := \mathcal{L}_{\text{box}}(\text{pred}, \text{true}) + \mathcal{L}_{\text{conf}}(\text{pred}, \text{true}) + \mathcal{L}_{\text{cls}}(\text{pred}, \text{true})
\end{aligned}$$
where $\text{pred}, \text{true} \in \mathbb{R}^{S} \oplus \mathbb{R}^{S} \oplus \mathbb{R}^{B \times 5 + C}$.

> Note that the loss function only penalizes classification error if an object is present in that grid cell (hence the conditional class probability discussed earlier). It also  nly penalizes bounding box coordinate error if that predictor is "responsible" for the ground truth box (i.e. has the highest IOU of any predictor in that grid cell).

Data augmentation:
> For data augmentation we introduce random scaling and translations of up to 20% of the original image size. We also randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space.


* problems with v1:
    * can only input images with the same size as the training samples.
    * each cell only predicts one object. when a cell contains multiple objects, it can only predict one of them.
    * the IOU loss for big objects and small objects are similar. the square root technique did not solve the problem completely.
