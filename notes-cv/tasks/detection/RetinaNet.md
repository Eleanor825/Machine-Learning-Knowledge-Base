# RetinaNet and Focal Loss

* RetinaNet is an one-stage detector.
* Previously, one-stage detectors, though are fast, they have lower precisions. This is believed to be due to the imbalance between positive and negative examples. 
* Usually the one-stage detectors will generate 10~100k bounding boxes and only a few of them are positive examples. This is extremely imbalanced.
* The Problem of Class Imbalance:
    * Imbalance between positive/negative. \
    Usually in an image, most of the parts are backgrounds and only a few small regions are objects. So there are a lot more negative examples than positive examples. During training, the loss due to the negative examples dominates the loss due to the positive ones. This does not help converging. The model can still achieve a high accuracy if it just blindly predicts all boxes as background simply because too many of them are background.
    * Imbalance between hard/easy. \
    We need more hard examples, either positive or negative. Most of the negative examples are not at the boundary between objects and the background. So most negative examples are easy examples. This causes a small loss and does not help converging.
* Focal Loss:
    * Used the focal loss to tackle the problem of imbalanced examples.
    * The weighted loss in YOLO and the hard examples mining in SSD can only solve the first problem but not the second. This is why focal loss is here.
    * The original cross-entropy loss:
    $$\mathcal{L}^{CE}(p, y) = \begin{cases}
        -\log(p),   & \text{ if } y = 1 \\
        -\log(1-p), & \text{ if } y = 0.
    \end{cases}$$
    Note that when $p = 0.5$, the loss is around $0.693$, which is quite big.
    * In order to balance the positive/negative examples, add the balance factor $\alpha$ to get the $\alpha$-balanced cross-entropy loss.
    $$\mathcal{L}^{BCE}(p,y; \alpha) = \begin{cases}
        -\alpha\log(p),       & \text{ if } y = 1 \\
        -(1-\alpha)\log(1-p), & \text{ if } y = 0.
    \end{cases}$$
    But this only deals with the imbalance between positive/negative samples, but not the hard/easy samples.
    * Introduce $\gamma$ to put less loss on the examples with high confidence $p$ (easy) and more loss on the examples with low $p$ (hard).
    $$\mathcal{L}^{F}(p,y; \alpha,\gamma) := \begin{cases}
        -\alpha(1-p)^{\gamma}\log(p),   & \text{ if } y = 1 \\
        -(1-\alpha)p^{\gamma}\log(1-p), & \text{ if } y = 0.
    \end{cases}$$
    * Parameter values in experiment: $\gamma = 2$ and $\alpha = 0.75$.
    Now the priority of the examples when training is: hard positive > hard negative > easy positive > easy negative.
    * Properties:
        * Decreases as $p$ increases.
* Stability of Training: The Prior $\pi$:
    * Initialize the model with a prior $\pi$.
* The Anchor Boxes:
    * The anchors boxes with an IoU greater that 0.5 will be labelled as positive and the ones with IoU in $(0,0.4)$ will be labelled as negative (background). The ones with IoU in $(0.4, 0.5)$ will be ignored.
