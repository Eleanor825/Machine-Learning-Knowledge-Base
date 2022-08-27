# ResNet

## Reference

* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Identity vs. Projection Shortcuts

Three Options for Increasing Dimensions:
* Zero-padding shortcuts are used for increasing dimensions. All shortcuts are parameter-free.
* Projection shortcuts are used for increasing dimensions. Other shortcuts are identity.
* All shortcuts are projection shortcuts.

All three options performed similarly.
So don't use the third one.

## Bottleneck Architecture

* Each residual block consists of 3 convolutional layers.
    * 1st layer: $1 \times 1$. Responsible for decreasing dimensions.
    * 2nd layer: $3 \times 3$.
    * 3rd layer: $1 \times 1$. Responsible for increasing dimensions.

The parameter-free identity shortcuts are particularly important for the bottleneck architectures.
If the identity shortcut is replaced with projection, one can show that the time complexity and model size are doubled,
as the shortcut is connected to the two high-dimensional ends.
