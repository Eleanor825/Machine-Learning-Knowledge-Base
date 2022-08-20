# [Notes][Vison] Common Loss Functions

## Segmentation

Notations:
* Let $C \in \mathbb{Z}_{++}$ denote the number of classes.
* Let $n_{ij} \in \mathbb{Z}_{++}$ denote the number of pixels of class $i$ that are predicted as class $j$, for $i, j \in \{1, ..., C\}$.

Then we define:
* Pixel Accuracy $\displaystyle := \sum_{i}n_{ii} / \sum_{ij}n_{ij}$.
* Mean Accuracy $\displaystyle := \frac{1}{C}\sum_{i}\frac{n_{ii}}{\sum_{j}n_{ij}}$.
* Mean IoU $\displaystyle := \frac{1}{C}\sum_{i}\frac{n_{ii}}{\sum_{j}n_{ij} + \sum_{j}n_{ji} - n_{ii}}$.
* Frequency weighted IoU $\displaystyle := \sum_{i}\frac{\sum_{j}n_{ij}}{\sum_{jk}n_{jk}}\frac{n_{ii}}{\sum_{j}n_{ij} + \sum_{j}n_{ji} - n_{ii}}$.
