# CNN Models

## LeNet-5 (1998)


## AlexNet (2012)
* Smaller Filters
    * Doing two consecutive $3 \times 3$ convolutional layers is similar to doing a single $7 \times 7$ layer in terms of receptive field.
    * However, two $3 \times 3$ kernels has parameters only $3 \times 3 \times 2 = 18$ and one $7 \times 7$ kernel has parameters $7 \times 7$. So the latter is more computationally expensive.
* Deeper Networks


## ZFNet (2013)
## VGGNet (2014)
## DenseNet
## GoogLeNet


## Fully Convolutional Networks (FCN)
* Image Level Task: \
Get information about the image as a whole.
Produces numerical description of the entire image.
* Pixel Level Task: 
* Accepts inputs of any size.
* FCN outputs an labelled image (pixels are classified).
