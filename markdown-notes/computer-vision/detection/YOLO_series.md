# YOLO Series

The idea is to treat the object detection task as a regression problem.
Applied weights on the loss for positive/negative examples to tackle the problem of imbalanced positive/negative examples.

## YOLO-v1 (2015)
    * confidence = Pr(Object) * IoU \
        confidence being 0 means there is no object.
        confidence being not 0 means the IoU of the predicted bounding box and the ground truth box.
    * the output of the network is of shape $S \times S \times (5 \times B + C)$.
    * multiply the class probabilities with the confidence of the bounding boxes to get the class-specific confidence scores for each bounding box.
    * The loss function:
        * add more weight to the 8-dimensional coordinates predictions.
        * give less weight to the confidence loss for boxes with no objects.
        * Formula:
        $$\lambda_{\text{coord}}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathbb{1}_{i,j}^{\text{obj}}\big[(x_{i}-\hat{x}_{i})^{2}+(y_{i}-\hat{y}_{i})^{2}\big]$$
        $$+\lambda_{\text{coord}}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathbb{1}_{i,j}^{\text{obj}}\big[(\sqrt{w_{i}}-\sqrt{\hat{w}_{i}})^{2}+(\sqrt{h_{i}}-\sqrt{\hat{h}_{i}})^{2}\big]$$
        $$+\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathbb{1}_{i,j}^{\text{obj}}(C_{i}-\hat{C}_{i})^{2}$$
        $$+\lambda_{\text{noobj}}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathbb{1}_{i,j}^{\text{noobj}}(C_{i}-\hat{C}_{i})^{2}$$
        $$+\sum_{i=0}^{S^{2}}\mathbb{1}_{i}^{\text{obj}}\sum_{c\in\text{classes}}(p_{i}(c)-\hat{p}_{i}(c))^{2}.$$
    * problems with v1:
        * can only input images with the same size as the training samples.
        * each cell only predicts one object. when a cell contains multiple objects, it can only predict one of them.
        * the IOU loss for big objects and small objects are similar. the square root technique did not solve the problem completely.

## YOLO-v2 (2016)
    * Anchor mechanism.
        * Key idea: predict the offsets. \
        Rather than predicting the locations of the bounding boxes directly, YOLO-v2 generates anchor boxes and predicts the offsets.
        * Generating the anchors:
            * Hand crafted.
            * Learn from the dataset: Do K-means clustering on the ground truth bounding boxes in the training set.
    Used K-means algorithm to generate learned anchors.
    Reshape feature map.

## YOLO-v3
    * A combination of feature pyramid and anchor mechanism.

## YOLO-v4
    * YOLOv4 = CSPDarknet53+SPP+PAN+YOLOv3
