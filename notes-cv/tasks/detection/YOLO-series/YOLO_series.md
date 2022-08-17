# YOLO Series

The idea is to treat the object detection task as a regression problem.
Applied weights on the loss for positive/negative examples to tackle the problem of imbalanced positive/negative examples.

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
