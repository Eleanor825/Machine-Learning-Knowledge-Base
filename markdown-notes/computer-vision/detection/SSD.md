<link rel="stylesheet" href="../../style.css">

# SSD: Single Shot MultiBox Detector

* A combination of feature pyramid and anchor mechanism.

## Default Boxes

* Default boxes are similar to anchor boxes but are applied to several feature maps of different resolutions.

### Choosing Scales

The scale ($s_{k}$) of the default boxes for each feature map is computed as:

$$s_{k} := s_{\text{min}} + \frac{s_{\text{max}}-s_{\text{min}}}{m-1}(k-1)$$
where $s_{\text{min}} = 0.2$ and $s_{\text{max}} = 0.9$.

### Choosing Aspect Ratios

$$a_{r} \in \{1, 2, 3, \frac{1}{2}, \frac{1}{3}\}.$$

## Loss Function

### Notations

$x_{ij}^{p} \in \{0,1\}$: indicator for matching the $i$-th default box to the $j$-th ground truth box of category $p$.

### Total Loss

The overall objective loss is a weighted sum of localization loss ($\mathcal{L}_{\text{loc}}$) and confidence loss ($\mathcal{L}_{\text{conf}}$):

$$\begin{equation*}\boxed{\mathcal{L}(x,c,l,g) := \frac{1}{N}(\mathcal{L}_{\text{conf}}(x,c) + \alpha\mathcal{L}_{\text{loc}}(x,l,g))}\end{equation*}$$
where $N$ is the number of matched default boxes.

If $N = 0$, then we set $\mathcal{L} = 0$.
The weight term $\alpha$ is set to 1 by cross validation.

### Localization Loss

A smooth L1 loss between the predicted box ($l$) and the ground truth box ($g$) parameters.
We regress the offsets for $cx$, $cy$, $w$, $h$ of the default bounding boxes ($d$).

$$\begin{equation*}\boxed{
    \mathcal{L}_{\text{loc}}(x,l,g) := \sum_{i\in\text{Pos}}^{N}\sum_{m\in\{cx,cy,w,h\}}x_{ij}^{k}\text{smooth}_{\text{L1}}(l_{i}^{m} - \hat{g}_{j}^{m})
    }\end{equation*}$$
where
$$
\hat{g}_{j}^{cx} := \frac{g_{j}^{cx} - d_{i}^{cx}}{d_{i}^{w}}, \quad
\hat{g}_{j}^{cy} := \frac{g_{j}^{cy} - d_{i}^{cy}}{d_{i}^{h}}, \quad
\hat{g}_{j}^{w}  := \log\frac{g_{j}^{w}}{d_{i}^{w}}, \quad
\hat{g}_{j}^{h}  := \log\frac{g_{j}^{h}}{d_{i}^{h}}.
$$

### Confidence Loss

A softmax loss over multiple class confidences ($c$).

$$\begin{equation*}\boxed{\mathcal{L}_{\text{conf}}(x,c) := -\sum_{i\in\text{Pos}}^{N}x_{ij}^{p}\log(\hat{c}_{i}^{p}) - \sum_{i\in\text{Neg}}\log(\hat{c}_{i}^{0})}\end{equation*}$$
where
$$\hat{c}_{i}^{p} := \frac{\exp(c_{i}^{p})}{\sum_{p}\exp(c_{i}^{p})}.$$

## Hard Negative Mining

* After the matching step, most of the default boxes are negatives, especially when the number of possible default boxes is large.
* This introduces a significant imbalance between the positive and negative training examples.
* Instead of using all the negative examples, we sort them using the highest confidence loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most $3:1$.
