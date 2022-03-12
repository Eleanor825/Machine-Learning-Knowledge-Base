# RCNNs

* R-CNN \
    Generate ~2k region proposals using Selective Search. Too slow.
* SPPNet \
    Combined classification and regression networks. Faster but still based on selective search to generate region proposals.
* Fast R-CNN \
    Multi-task loss function.


## Faster R-CNN (2015)

* Problems solved:
    * Region proposals generation too slow: \
    Generate region proposals using CNN (Region Proposal Network).
    * Imbalanced positive/negative examples: \
    Balance the positive/negative examples to 1:3 before passing them to the Region of Interest Pooling (RoI Pooling) layer.
    * Too many easy negative: \
    Filter out the examples with a high probability of being a foreground when doing the FPN.
* Anchor mechanism.
