# [Notes][Vision][Segmentation] Panoptic Segmentation

* url: https://arxiv.org/abs/1801.00868
* Title: Panoptic Segmentation
* Year: 03 Jan `2018`
* Authors: Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, Piotr Dollár
* Institutions: [Facebook AI Research (FAIR)], [HCI/IWR, Heidelberg University, Germany]
* Abstract: We propose and study a task we name panoptic segmentation (PS). Panoptic segmentation unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance). The proposed task requires generating a coherent scene segmentation that is rich and complete, an important step toward real-world vision systems. While early work in computer vision addressed related image/scene parsing tasks, these are not currently popular, possibly due to lack of appropriate metrics or associated recognition challenges. To address this, we propose a novel panoptic quality (PQ) metric that captures performance for all classes (stuff and things) in an interpretable and unified manner. Using the proposed metric, we perform a rigorous study of both human and machine performance for PS on three existing datasets, revealing interesting insights about the task. The aim of our work is to revive the interest of the community in a more unified view of image segmentation.

----------------------------------------------------------------------------------------------------

## 1. Introduction

> In the early days of computer vision, `things` – `countable objects` such as people, animals, tools – received the dominant share of attention. Questioning the wisdom of this trend, Adelson [1] elevated the importance of studying systems that recognize `stuff` – `amorphous regions` of similar texture or material such as grass, sky, road.

> Studying stuff is most commonly formulated as a task known as semantic segmentation, see Figure 1. As stuff is amorphous and uncountable, this task is defined as simply assigning a category label to each pixel in an image (note that semantic segmentation treats thing categories as stuff).

> In contrast, studying things is typically formulated as the task of object detection or instance segmentation, where the goal is to detect each object and delineate it with a bounding box or segmentation mask, respectively.

> The schism between semantic and instance segmentation has led to a parallel rift in the methods for these tasks. Stuff classifiers are usually built on fully convolutional nets [26] with dilations [41, 5] while object detectors often use object proposals [15] and are region-based [33, 14].

> With these questions in mind, we propose a new task that encompasses both stuff and things. We refer to the resulting task as Panoptic Segmentation (PS). The definition of panoptic is "including everything visible in one view", in our context panoptic refers to a unified, global view of segmentation. The task formulation of PS is deceptively simple: each pixel of an image must be assigned a semantic label and an instance id.

> Panoptic segmentation is a generalization of both semantic and instance segmentation but introduces new algorithmic challenges. Unlike semantic segmentation, PS requires differentiating individual object instances; this poses a challenge for fully convolutional nets. Unlike instance segmentation, in PS object segments must be non-overlapping; this presents a challenge for region-based methods that operate on each object independently.

## 2. Related Work



----------------------------------------------------------------------------------------------------

## References

* Kirillov, Alexander, et al. "Panoptic segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2019.

## Further Reading

* [5] DeepLabv2
* [26] FCN
* [41] Dilated Convolutions
