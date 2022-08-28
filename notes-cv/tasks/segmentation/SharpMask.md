# [Notes][Vision][Segmentation] SharpMask

* url: https://arxiv.org/abs/1603.08695
* Title: Learning to Refine Object Segments
* Year: 29 Mar `2016`
* Authors: Pedro O. Pinheiro, Tsung-Yi Lin, Ronan Collobert, Piotr Dollàr
* Institutions: [Facebook AI Research (FAIR)]
* Abstract: Object segmentation requires both object-level information and low-level pixel data. This presents a challenge for feedforward networks: lower layers in convolutional nets capture rich spatial information, while upper layers encode object-level knowledge but are invariant to factors such as pose and appearance. In this work we propose to augment feedforward nets for object segmentation with a novel top-down refinement approach. The resulting bottom-up/top-down architecture is capable of efficiently generating high-fidelity object masks. Similarly to skip connections, our approach leverages features at all layers of the net. Unlike skip connections, our approach does not attempt to output independent predictions at each layer. Instead, we first output a coarse `mask encoding' in a feedforward pass, then refine this mask encoding in a top-down pass utilizing features at successively lower layers. The approach is simple, fast, and effective. Building on the recent DeepMask network for generating object proposals, we show accuracy improvements of 10-20% in average recall for various setups. Additionally, by optimizing the overall network architecture, our approach, which we call SharpMask, is 50% faster than the original DeepMask network (under .8s per image).

----------------------------------------------------------------------------------------------------

## 1 Introduction

> In this paper, we propose a novel CNN which efficiently merges the spatially rich information from low-level features with the high-level object knowledge encoded in upper network layers. Rather than generating independent outputs from multiple network layers, our approach first generates a coarse mask encoding in a feedforward manner, which is simply a semantically meaningful feature map with multiple channels, then refines it by successively integrating information from earlier layers.

> Specifically, we introduce a `refinement module` and stack successive such modules together into a top-down refinement process.

> Each refinement module is responsible for ‘inverting’ the effect of pooling by taking a mask encoding generated in the top-down pass, along with the matching features from the bottom-up pass, and merging the information in both to generate a new mask encoding with double the spatial resolution.

> In this work we utilize the DeepMask architecture as our starting point for object instance segmentation due to its simplicity and effectiveness.

## 2 Related Work

## 3 Learning Mask Refinement

### 3.1 Refinement Overview

> Our goal is to efficiently merge the `spatially` rich information from low-level features with the high-level `semantic` information encoded in upper network layers.

> Three principles guide our approach:
> 1. object-level information is often necessary to segment an object,
> 2. given object-level information, segmentation should proceed in a top-down fashion, successively integrating information from earlier layers, and
> 3. the approach should invert the loss of resolution from pooling (with the final output matching the resolution of the input).

### 3.2 Refinement Details

Notations:
* Let $H, W \in \mathbb{Z}_{++}$ denote the height and width of the input image.
* Let $H^{(i)}, W^{(i)} \in \mathbb{Z}_{++}$ denote the height and width of the $i$-th feature map and mask encoding.
* Let $k_{m}^{(i)} \in \mathbb{Z}_{++}$ denote the number of channels of the $i$-th mask encoding.
* Let $M^{(i)} \in \mathbb{R}^{H^{(i)} \times W^{(i)} \times k_{m}^{(i)}}$ denote the $i$-th mask encoding.
* Let $k_{f}^{(i)} \in \mathbb{Z}_{++}$ denote the number of channels of the $i$-th feature map.
* Let $F^{(i)} \in \mathbb{R}^{H^{(i)} \times W^{(i)} \times k_{f}^{(i)}}$ denote the $i$-th feature map.
* Let $k_{s}^{(i)} \in \mathbb{Z}_{++}$ denote the number of channels of the $i$-th skip feature map.
* Let $S^{(i)} \in \mathbb{R}^{H^{(i)} \times W^{(i)} \times k_{s}^{(i)}}$ denote the $i$-th skip feature map.
* Let $R^{(i)}: \mathbb{R}^{H^{(i)} \times W^{(i)} \times k_{m}^{(i)}} \oplus \mathbb{R}^{H^{(i)} \times W^{(i)} \times k_{f}^{(i)}} \to \mathbb{R}^{H^{(i+1)} \times W^{(i+1)} \times k_{m}^{(i+1)}}$ denote the $i$-th refinement module.

Then
$$\begin{aligned}
    S^{(i)} & := \operatorname{ReLU}(\operatorname{Conv}(F^{(i)})) \\
    R^{(i)}(F^{(i)}, M^{(i)}) & := \operatorname{upsample}(\operatorname{ReLU}(\operatorname{Conv}(\operatorname{Concat}(S^{(i)}, M^{(i)})))) \\
    M^{(i+1)} & := R^{(i)}(F^{(i)}, M^{(i)}).
\end{aligned}$$

> As with the convolution for generating the skip features, this transformation is used to simultaneously learn a nonlinear mask encoding from the concatenated features and to control the capacity of the model.

> Note that the refinement module uses only convolution, ReLU, bilinear upsampling, and concatenation, hence it is fully backpropable and highly efficient.

> As a general design principle, we aim to keep $k_{s}^{(i)}$ and $k_{m}^{(i)}$ large enough to capture rich information but small enough to keep computation low.

> In particular, we can start with a fairly large number of channels but as spatial resolution is increased the number of channels should decrease. This reverses the typical design of feedforward networks where spatial resolution decreases while the number of channels increases with increasing depth.

### 3.3 Training and Inference

Training

> Training proceeds in two stages: first, the model is trained to jointly infer a coarse pixel-wise segmentation mask and an object score, second, the feedforward path is ‘frozen’ and the refinement modules trained.

> The first training stage is identical to [22].

> Once learning of the first stage converges, the final mask prediction layer of the feedforward network is removed and replaced with a linear layer that generates a mask encoding $M^{(1)}$ in place of the actual mask output. We then add the refinement modules to the network and train using standard stochastic gradient descent, backpropagating the error only on the horizontal and vertical convolution layers on each of the $n$ refinement modules.

> This two-stage training procedure was selected for three reasons.
> 1. we found it led to faster convergence.
> 2. at inference time, a single network trained in this manner can be used to generate either a coarse mask using the forward path only or a sharp mask using our bottom-up/top-down approach.
> 3. we found the gains of fine-tuning through the entire network to be minimal once the forward branch had converged.

Inference

> During full-image inference, similarly to [22], most computation for neighboring windows is shared through use of convolution, including for skip layers $S^{(i)}$.

> However, as discussed, the refinement modules receive a unique input $M^{(1)}$ at each spatial location, hence, computation proceeds independently at each location for this stage. Rather than refine every proposal, we simply refine only the most promising locations. Specifically, we select the top $N$ scoring proposal windows and apply the refinement in a batch mode to these top $N$ locations.

## 4 Feedforward Architecture

## 5 Experiments

## 6 Conclusion

----------------------------------------------------------------------------------------------------

## References

## Further Reading

* [1] DPM
* [2] OverFeat
* [3] MSC-MultiBox
* [4] Spatial Pyramid Pooling (SPP)
* [5] R-CNN
* [6] Fast R-CNN
* [7] Faster R-CNN
* [8] Inside-Outside Net (ION)
* [17] DeepLabv1
* [19] [Learning Deconvolution](https://zhuanlan.zhihu.com/p/558646271)
* [22] DeepMask
* [24] Hypercolumns
* [25] AlexNet
* [26] VGG
* [27] Inception-v1/GoogLeNet
* [28] ResNet
* [29] Fully Convolutional Networks (FCN)
