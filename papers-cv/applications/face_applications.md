# [Papers][Vision] Face Detection and Recognition <!-- omit in toc -->

count=6

## Table of Contents <!-- omit in toc -->

- [Detection](#detection)
  - [Unknown](#unknown)
  - [Knowledge Based](#knowledge-based)
  - [Feature Based](#feature-based)
  - [Appearance Based](#appearance-based)
  - [Template Matching](#template-matching)
- [Recognition](#recognition)

----------------------------------------------------------------------------------------------------

## Detection

### Unknown

* [[DSFD](https://arxiv.org/abs/1810.10220)]
    [[pdf](https://arxiv.org/pdf/1810.10220)]
    * Title: DSFD: Dual Shot Face Detector
    * Year: 24 Oct `2018`
    * Authors: Jian Li, Yabiao Wang, Changan Wang, Ying Tai, Jianjun Qian, Jian Yang, Chengjie Wang, Jilin Li, Feiyue Huang
    * Institutions: [School of Computer Science and Engineering, Nanjing University of Science and Technology], [Youtu Lab, Tencent]
    * Abstract: In this paper, we propose a novel face detection network with three novel contributions that address three key aspects of face detection, including better feature learning, progressive loss design and anchor assign based data augmentation, respectively. First, we propose a Feature Enhance Module (FEM) for enhancing the original feature maps to extend the single shot detector to dual shot detector. Second, we adopt Progressive Anchor Loss (PAL) computed by two different sets of anchors to effectively facilitate the features. Third, we use an Improved Anchor Matching (IAM) by integrating novel anchor assign strategy into data augmentation to provide better initialization for the regressor. Since these techniques are all related to the two-stream design, we name the proposed network as Dual Shot Face Detector (DSFD). Extensive experiments on popular benchmarks, WIDER FACE and FDDB, demonstrate the superiority of DSFD over the state-of-the-art face detectors.
* [[RetinaFace](https://ieeexplore.ieee.org/document/9157330)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9157330)]
    * Title: RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild
    * Year: 05 August `2020`
    * Authors: Jiankang Deng; Jia Guo; Evangelos Ververas; Irene Kotsia; Stefanos Zafeiriou
    * Institutions: 
    * Abstract: Though tremendous strides have been made in uncontrolled face detection, accurate and efficient 2D face alignment and 3D face reconstruction in-the-wild remain an open challenge. In this paper, we present a novel single-shot, multi-level face localisation method, named RetinaFace, which unifies face box prediction, 2D facial landmark localisation and 3D vertices regression under one common target: point regression on the image plane. To fill the data gap, we manually annotated five facial landmarks on the WIDER FACE dataset and employed a semi-automatic annotation pipeline to generate 3D vertices for face images from the WIDER FACE, AFLW and FDDB datasets. Based on extra annotations, we propose a mutually beneficial regression target for 3D face reconstruction, that is predicting 3D vertices projected on the image plane constrained by a common 3D topology. The proposed 3D face reconstruction branch can be easily incorporated, without any optimisation difficulty, in parallel with the existing box and 2D landmark regression branches during joint training. Extensive experimental results show that RetinaFace can simultaneously achieve stable face detection, accurate 2D face alignment and robust 3D face reconstruction while being efficient through single-shot inference.
* [[BlazeFace](https://arxiv.org/abs/1907.05047)]
    [[pdf](https://arxiv.org/pdf/1907.05047)]
    * Title: BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs
    * Year: 11 Jul `2019`
    * Authors: Valentin Bazarevsky, Yury Kartynnik, Andrey Vakunov, Karthik Raveendran, Matthias Grundmann
    * Institutions: [Google Research]
    * Abstract: We present BlazeFace, a lightweight and well-performing face detector tailored for mobile GPU inference. It runs at a speed of 200-1000+ FPS on flagship devices. This super-realtime performance enables it to be applied to any augmented reality pipeline that requires an accurate facial region of interest as an input for task-specific models, such as 2D/3D facial keypoint or geometry estimation, facial features or expression classification, and face region segmentation. Our contributions include a lightweight feature extraction network inspired by, but distinct from MobileNetV1/V2, a GPU-friendly anchor scheme modified from Single Shot MultiBox Detector (SSD), and an improved tie resolution strategy alternative to non-maximum suppression.

### Knowledge Based

### Feature Based

### Appearance Based

### Template Matching

## Recognition

* [[FaceNet](https://arxiv.org/abs/1503.03832)]
    [[pdf](https://arxiv.org/pdf/1503.03832.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1503.03832/)]
    * Title: FaceNet: A Unified Embedding for Face Recognition and Clustering
    * Year: 12 Mar `2015`
    * Authors: Florian Schroff, Dmitry Kalenichenko, James Philbin
    * Institutions: [Google Inc.]
    * Abstract: Despite significant recent advances in the field of face recognition, implementing face verification and recognition efficiently at scale presents serious challenges to current approaches. In this paper we present a system, called FaceNet, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors. Our method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches. To train, we use triplets of roughly aligned matching / non-matching face patches generated using a novel online triplet mining method. The benefit of our approach is much greater representational efficiency: we achieve state-of-the-art face recognition performance using only 128-bytes per face. On the widely used Labeled Faces in the Wild (LFW) dataset, our system achieves a new record accuracy of 99.63%. On YouTube Faces DB it achieves 95.12%. Our system cuts the error rate in comparison to the best published result by 30% on both datasets. We also introduce the concept of harmonic embeddings, and a harmonic triplet loss, which describe different versions of face embeddings (produced by different networks) that are compatible to each other and allow for direct comparison between each other.
* [[DeepFace](https://ieeexplore.ieee.org/document/6909616)]
    [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6909616)]
    * Title: DeepFace: Closing the Gap to Human-Level Performance in Face Verification
    * Year: 25 September `2014`
    * Authors: Yaniv Taigman; Ming Yang; Marc'Aurelio Ranzato; Lior Wolf
    * Institutions: [Facebook AI Research], [Tel Aviv University]
    * Abstract: In modern face recognition, the conventional pipeline consists of four stages: detect => align => represent => classify. We revisit both the alignment step and the representation step by employing explicit 3D face modeling in order to apply a piecewise affine transformation, and derive a face representation from a nine-layer deep neural network. This deep network involves more than 120 million parameters using several locally connected layers without weight sharing, rather than the standard convolutional layers. Thus we trained it on the largest facial dataset to-date, an identity labeled dataset of four million facial images belonging to more than 4, 000 identities. The learned representations coupling the accurate model-based alignment with the large facial database generalize remarkably well to faces in unconstrained environments, even with a simple classifier. Our method reaches an accuracy of 97.35% on the Labeled Faces in the Wild (LFW) dataset, reducing the error of the current state of the art by more than 27%, closely approaching human-level performance.
* [[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)]
    [[pdf](https://arxiv.org/pdf/1604.02878)]
    * Title: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
    * Year: 11 Apr `2016`
    * Authors: Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao
    * Institutions: 
    * Abstract: Face detection and alignment in unconstrained environment are challenging due to various poses, illuminations and occlusions. Recent studies show that deep learning approaches can achieve impressive performance on these two tasks. In this paper, we propose a deep cascaded multi-task framework which exploits the inherent correlation between them to boost up their performance. In particular, our framework adopts a cascaded structure with three stages of carefully designed deep convolutional networks that predict face and landmark location in a coarse-to-fine manner. In addition, in the learning process, we propose a new online hard sample mining strategy that can improve the performance automatically without manual sample selection. Our method achieves superior accuracy over the state-of-the-art techniques on the challenging FDDB and WIDER FACE benchmark for face detection, and AFLW benchmark for face alignment, while keeps real time performance.
