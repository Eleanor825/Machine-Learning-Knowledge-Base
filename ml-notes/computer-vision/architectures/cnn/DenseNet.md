# DenseNet

## Reference

* [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
* [Code for the Original Paper](https://github.com/liuzhuang13/DenseNet)

## Notes

* Growth Rate
    * The $l$-th layer has $k_{0} + k \times (l-1)$ input feature maps.
    * DenseNet can have very narrow layers, e.g., $k = 12$.
    * One exxplanation for this is that each layer has access to all the preceding feature-maps in its block and, therefore, to the network's "collective knowledge".
    * One can view the feature-maps as the global state of the network.
* Variants
    * DenseNet-B: Bottleneck layers
        * Improve computational efficiency.
        * BN-ReLU-Conv($1 \times 1$)-BN-ReLU-Conv($3 \times 3$)
    * DenseNet-C: Compression
        * Improve model compactness.
        * If a dense block contains $m$ feature-maps, we let the following transition layer generate $\lfloor \theta m \rfloor$ outut feature-maps.
* Parameter Efficiency
    * DenseNet-BC is consistently the most parameter efficient variant of DenseNet.
    * A DenseNet-BC with only 0.8M trainable parameters is able to achieve comparable accuracy as the 1001-layer (pre-activation) ResNet with 10.2M parameters.
