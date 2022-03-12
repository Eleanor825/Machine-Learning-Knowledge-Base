# MobileNet - Google

## MobileNet v1 (2017)
    * Depthwise Separable Convolution: \
    Separated convolution operation into a depth-wise convolution and a point-wise convolution to reduce computations.

## MobileNet v2 (2018)
    * Linear Bottleneck: \
    Replaced the last layers of ReLU with linear activations.
    * Expansion Layer: \
    Expand the depth using 1 by 1 convolution.
    * Inverted Residual Block:
        * ResNet: 0.25 times dimension decrease -> 3 by 3 convolution -> dimension increase
        * MobileNet v2: 6 times dimension increase -> depth-wise separable convolution -> dimension decrease

## MobileNet v3 (2019)
    * NAS:
    * Squeeze and Excitation:
    * h-swish activation function:
        * swish activation function:
        $$\operatorname{swish}(x) = x \cdot \operatorname{sigmoid}(\beta x).$$
        * h-swish activation function: \
        aims to approximate the swish activation function since it is slow.
        $$\operatorname{h-swish}(x) = x \cdot \frac{1}{6}\operatorname{ReLU}6(x+3).$$
    * did something to the tail of v2.
