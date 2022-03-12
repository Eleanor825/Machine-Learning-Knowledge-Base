## True/False Positive/Negative
* True Positive (TP): a correct detection (IoU > threshold)
* False Positive (FP): a wrong detection (IoU < threshold)
* False Negative (FN): a ground truth not detected
* True Negative (TN): NA.


## Precision and Recall
* Precision is the ability of a model to identify only the relevant objects. It is the percentage of correct positive predictions.
$$\text{precision} = \frac{\text{TP}}{\text{all detections}} = \frac{\text{TP}}{\text{TP}+\text{FP}}.$$
* Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths.
$$\text{recall} = \frac{\text{TP}}{\text{all ground truths}} = \frac{\text{TP}}{\text{TP}+\text{FN}}.$$


## Metrics
* Precision-Recall Curve:
    * An object detector of a particular class is considered good if its precision stays high as recall increases, which means that if you vary the confidence threshold, the precision and recall will still be high.
* Average Precision (AP):
    * AP is the area under the precision-recall curve.
    * Mean average precision (mAP) is the average of AP across all classes.
