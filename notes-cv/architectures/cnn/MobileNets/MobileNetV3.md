# [MobileNetV3](https://arxiv.org/abs/1905.02244)

* Title: Searching for MobileNetV3
* Year: 06 May `2019`
* Author: Andrew Howard
* Abstract: We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2\% more accurate on ImageNet classification while reducing latency by 15\% compared to MobileNetV2. MobileNetV3-Small is 4.6\% more accurate while reducing latency by 5\% compared to MobileNetV2. MobileNetV3-Large detection is 25\% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30\% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.

----------------------------------------------------------------------------------------------------

* NAS:
* Squeeze and Excitation:
* h-swish activation function:
    * swish activation function:
    $$\operatorname{swish}(x) = x \cdot \operatorname{sigmoid}(\beta x).$$
    * h-swish activation function: \
    aims to approximate the swish activation function since it is slow.
    $$\operatorname{h-swish}(x) = x \cdot \frac{1}{6}\operatorname{ReLU}6(x+3).$$
* did something to the tail of v2.
