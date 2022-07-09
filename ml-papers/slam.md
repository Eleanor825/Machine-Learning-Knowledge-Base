<span style="font-family:monospace">

# Papers in Computer Vision - SLAM

count: 15

## Overviews

### Classical Age (1986-2004)

* [Simultaneous localization and mapping: part I](https://ieeexplore.ieee.org/document/1638022)
    * Title: Simultaneous localization and mapping: part I
    * Year: `2006`
    * Author: H. Durrant-Whyte
    * Abstract: This paper describes the simultaneous localization and mapping (SLAM) problem and the essential methods for solving the SLAM problem and summarizes key implementations and demonstrations of the method. While there are still many practical issues to overcome, especially in more complex outdoor environments, the general SLAM method is now a well understood and established part of robotics. Another part of the tutorial summarized more recent works in addressing some of the remaining issues in SLAM, including computation, feature representation, and data association.
* [Simultaneous localization and mapping (SLAM): part II](https://ieeexplore.ieee.org/document/1678144)
    * Title: Simultaneous localization and mapping (SLAM): part II
    * Year: `2006`
    * Author: T. Bailey
    * Abstract: This paper discusses the recursive Bayesian formulation of the simultaneous localization and mapping (SLAM) problem in which probability distributions or estimates of absolute or relative locations of landmarks and vehicle pose are obtained. The paper focuses on three key areas: computational complexity; data association; and environment representation.

### Algorithmic Analysis Age (2004-2015)

* [A review of recent developments in Simultaneous Localization and Mapping](https://ieeexplore.ieee.org/abstract/document/6038117)
    * Title: A review of recent developments in Simultaneous Localization and Mapping
    * Year: `2011`
    * Author: Gamini Dissanayake
    * Abstract: Simultaneous Localization and Mapping (SLAM) problem has been an active area of research in robotics for more than a decade. Many fundamental and practical aspects of SLAM have been addressed and some impressive practical solutions have been demonstrated. The aim of this paper is to provide a review of the current state of the research on feature based SLAM, in particular to examine the current understanding of the fundamental properties of the SLAM problem and associated issues with the view to consolidate recent achievements.
* [Visual Place Recognition: A Survey](https://ieeexplore.ieee.org/document/7339473)
    * Title: Visual Place Recognition: A Survey
    * Year: `2016`
    * Author: Stephanie Lowry
    * Abstract: Visual place recognition is a challenging problem due to the vast range of ways in which the appearance of real-world places can vary. In recent years, improvements in visual sensing capabilities, an ever-increasing focus on long-term mobile robot autonomy, and the ability to draw on state-of-the-art research in other disciplines-particularly recognition in computer vision and animal navigation in neuroscience-have all contributed to significant advances in visual place recognition systems. This paper presents a survey of the visual place recognition research landscape. We start by introducing the concepts behind place recognition-the role of place recognition in the animal kingdom, how a “place” is defined in a robotics context, and the major components of a place recognition system. Long-term robot operations have revealed that changing appearance can be a significant factor in visual place recognition failure; therefore, we discuss how place recognition solutions can implicitly or explicitly account for appearance change within the environment. Finally, we close with a discussion on the future of visual place recognition, in particular with respect to the rapid advances being made in the related fields of deep learning, semantic scene understanding, and video description.

### Robust Perception Age

* [Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age](https://arxiv.org/abs/1606.05830)
    * Title: Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age
    * Year: 19 Jun `2016`
    * Author: Cesar Cadena
    * Abstract: Simultaneous Localization and Mapping (SLAM)consists in the concurrent construction of a model of the environment (the map), and the estimation of the state of the robot moving within it. The SLAM community has made astonishing progress over the last 30 years, enabling large-scale real-world applications, and witnessing a steady transition of this technology to industry. We survey the current state of SLAM. We start by presenting what is now the de-facto standard formulation for SLAM. We then review related work, covering a broad set of topics including robustness and scalability in long-term mapping, metric and semantic representations for mapping, theoretical performance guarantees, active SLAM and exploration, and other new frontiers. This paper simultaneously serves as a position paper and tutorial to those who are users of SLAM. By looking at the published research with a critical eye, we delineate open challenges and new research issues, that still deserve careful scientific investigation. The paper also contains the authors' take on two questions that often animate discussions during robotics conferences: Do robots need SLAM? and Is SLAM solved?

## Others

* [SVO](https://ieeexplore.ieee.org/document/6906584)
    * Title: SVO: Fast semi-direct monocular visual odometry
    * Year: `2014`
    * Author: Christian Forster
    * Abstract: We propose a semi-direct monocular visual odometry algorithm that is precise, robust, and faster than current state-of-the-art methods. The semi-direct approach eliminates the need of costly feature extraction and robust matching techniques for motion estimation. Our algorithm operates directly on pixel intensities, which results in subpixel precision at high frame-rates. A probabilistic mapping method that explicitly models outlier measurements is used to estimate 3D points, which results in fewer outliers and more reliable points. Precise and high frame-rate motion estimation brings increased robustness in scenes of little, repetitive, and high-frequency texture. The algorithm is applied to micro-aerial-vehicle state-estimation in GPS-denied environments and runs at 55 frames per second on the onboard embedded computer and at more than 300 frames per second on a consumer laptop. We call our approach SVO (Semi-direct Visual Odometry) and release our implementation as open-source software.
* [DSO](https://arxiv.org/abs/1607.02565)
    * Title: Direct Sparse Odometry
    * Year: 09 Jul `2016`
    * Author: Jakob Engel
    * Abstract: We propose a novel direct sparse visual odometry formulation. It combines a fully direct probabilistic model (minimizing a photometric error) with consistent, joint optimization of all model parameters, including geometry -- represented as inverse depth in a reference frame -- and camera motion. This is achieved in real time by omitting the smoothness prior used in other direct methods and instead sampling pixels evenly throughout the images. Since our method does not depend on keypoint detectors or descriptors, it can naturally sample pixels from across all image regions that have intensity gradient, including edges or smooth intensity variations on mostly white walls. The proposed model integrates a full photometric calibration, accounting for exposure time, lens vignetting, and non-linear response functions. We thoroughly evaluate our method on three different datasets comprising several hours of video. The experiments show that the presented approach significantly outperforms state-of-the-art direct and indirect methods in a variety of real-world settings, both in terms of tracking accuracy and robustness.
* [LDSO](https://arxiv.org/abs/1808.01111)
    * Title: LDSO: Direct Sparse Odometry with Loop Closure
    * Year: 03 Aug `2018`
    * Author: Xiang Gao
    * Abstract: In this paper we present an extension of Direct Sparse Odometry (DSO) to a monocular visual SLAM system with loop closure detection and pose-graph optimization (LDSO). As a direct technique, DSO can utilize any image pixel with sufficient intensity gradient, which makes it robust even in featureless areas. LDSO retains this robustness, while at the same time ensuring repeatability of some of these points by favoring corner features in the tracking frontend. This repeatability allows to reliably detect loop closure candidates with a conventional feature-based bag-of-words (BoW) approach. Loop closure candidates are verified geometrically and Sim(3) relative pose constraints are estimated by jointly minimizing 2D and 3D geometric error terms. These constraints are fused with a co-visibility graph of relative poses extracted from DSO's sliding window optimization. Our evaluation on publicly available datasets demonstrates that the modified point selection strategy retains the tracking accuracy and robustness, and the integrated pose-graph optimization significantly reduces the accumulated rotation-, translation- and scale-drift, resulting in an overall performance comparable to state-of-the-art feature-based systems, even without global bundle adjustment.

## Localize the camera globally against the current map

* [Parallel Tracking and Mapping for Small AR Workspaces](https://ieeexplore.ieee.org/document/4538852)
    * Title: Parallel Tracking and Mapping for Small AR Workspaces
    * Year: `2007`
    * Author: Georg Klein
    * Abstract: This paper presents a method of estimating camera pose in an unknown scene. While this has previously been attempted by adapting SLAM algorithms developed for robotic exploration, we propose a system specifically designed to track a hand-held camera in a small AR workspace. We propose to split tracking and mapping into two separate tasks, processed in parallel threads on a dual-core computer: one thread deals with the task of robustly tracking erratic hand-held motion, while the other produces a 3D map of point features from previously observed video frames. This allows the use of computationally expensive batch optimisation techniques not usually associated with real-time operation: The result is a system that produces detailed maps with thousands of landmarks which can be tracked at frame-rate, with an accuracy and robustness rivalling that of state-of-the-art model-based systems.
* [ORB-SLAM](https://arxiv.org/abs/1502.00956)
    * Title: ORB-SLAM: a Versatile and Accurate Monocular SLAM System
    * Year: 03 Feb `2015`
    * Author: Raul Mur-Artal
    * Abstract: This paper presents ORB-SLAM, a feature-based monocular SLAM system that operates in real time, in small and large, indoor and outdoor environments. The system is robust to severe motion clutter, allows wide baseline loop closing and relocalization, and includes full automatic initialization. Building on excellent algorithms of recent years, we designed from scratch a novel system that uses the same features for all SLAM tasks: tracking, mapping, relocalization, and loop closing. A survival of the fittest strategy that selects the points and keyframes of the reconstruction leads to excellent robustness and generates a compact and trackable map that only grows if the scene content changes, allowing lifelong operation. We present an exhaustive evaluation in 27 sequences from the most popular datasets. ORB-SLAM achieves unprecedented performance with respect to other state-of-the-art monocular SLAM approaches. For the benefit of the community, we make the source code public.

## Track the camera locally with visual (keyframe) odometry

* [Dense visual SLAM for RGB-D cameras](https://ieeexplore.ieee.org/document/6696650)
    * Title: Dense visual SLAM for RGB-D cameras
    * Year: `2013`
    * Author: Christian Kerl
    * Abstract: In this paper, we propose a dense visual SLAM method for RGB-D cameras that minimizes both the photometric and the depth error over all pixels. In contrast to sparse, feature-based methods, this allows us to better exploit the available information in the image data which leads to higher pose accuracy. Furthermore, we propose an entropy-based similarity measure for keyframe selection and loop closure detection. From all successful matches, we build up a graph that we optimize using the g2o framework. We evaluated our approach extensively on publicly available benchmark datasets, and found that it performs well in scenes with low texture as well as low structure. In direct comparison to several state-of-the-art methods, our approach yields a significantly lower trajectory error. We release our software as open-source.

## Combination of both

* [VINS-Mono](https://arxiv.org/abs/1708.03852)
    * Title: VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator
    * Year: 13 Aug `2017`
    * Author: Tong Qin
    * Abstract: A monocular visual-inertial system (VINS), consisting of a camera and a low-cost inertial measurement unit (IMU), forms the minimum sensor suite for metric six degrees-of-freedom (DOF) state estimation. However, the lack of direct distance measurement poses significant challenges in terms of IMU processing, estimator initialization, extrinsic calibration, and nonlinear optimization. In this work, we present VINS-Mono: a robust and versatile monocular visual-inertial state estimator.Our approach starts with a robust procedure for estimator initialization and failure recovery. A tightly-coupled, nonlinear optimization-based method is used to obtain high accuracy visual-inertial odometry by fusing pre-integrated IMU measurements and feature observations. A loop detection module, in combination with our tightly-coupled formulation, enables relocalization with minimum computation overhead.We additionally perform four degrees-of-freedom pose graph optimization to enforce global consistency. We validate the performance of our system on public datasets and real-world experiments and compare against other state-of-the-art algorithms. We also perform onboard closed-loop autonomous flight on the MAV platform and port the algorithm to an iOS-based demonstration. We highlight that the proposed work is a reliable, complete, and versatile system that is applicable for different applications that require high accuracy localization. We open source our implementations for both PCs and iOS mobile devices.

## Loop Detection Methods

* [A visual bag of words method for interactive qualitative localization and mapping](https://ieeexplore.ieee.org/document/4209698)
    * Title: A visual bag of words method for interactive qualitative localization and mapping
    * Year: `2007`
    * Author: David Filliat
    * Abstract: Localization for low cost humanoid or animal-like personal robots has to rely on cheap sensors and has to be robust to user manipulations of the robot. We present a visual localization and map-learning system that relies on vision only and that is able to incrementally learn to recognize the different rooms of an apartment from any robot position. This system is inspired by visual categorization algorithms called bag of words methods that we modified to make fully incremental and to allow a user-interactive training. Our system is able to reliably recognize the room in which the robot is after a short training time and is stable for long term use. Empirical validation on a real robot and on an image database acquired in real environments are presented.
* [FAB-MAP](https://journals.sagepub.com/doi/abs/10.1177/0278364908090961)
    * Title: FAB-MAP: Probabilistic Localization and Mapping in the Space of Appearance
    * Year: `2008`
    * Author: Mark Cummins
    * Abstract: This paper describes a probabilistic approach to the problem of recognizing places based on their appearance. The system we present is not limited to localization, but can determine that a new observation comes from a previously unseen place, and so augment its map. Effectively this is a SLAM system in the space of appearance. Our probabilistic approach allows us to explicitly account for perceptual aliasing in the environment—identical but indistinctive observations receive a low probability of having come from the same place. We achieve this by learning a generative model of place appearance. By partitioning the learning problem into two parts, new place models can be learned online from only a single observation of a place. The algorithm complexity is linear in the number of places in the map, and is particularly suitable for online loop closure detection in mobile robotics.
* [Bags of Binary Words for Fast Place Recognition in Image Sequences](https://ieeexplore.ieee.org/document/6202705)
    * Title: Bags of Binary Words for Fast Place Recognition in Image Sequences
    * Year: `2012`
    * Author: Dorian Galvez-López
    * Abstract: We propose a novel method for visual place recognition using bag of words obtained from accelerated segment test (FAST)+BRIEF features. For the first time, we build a vocabulary tree that discretizes a binary descriptor space and use the tree to speed up correspondences for geometrical verification. We present competitive results with no false positives in very different datasets, using exactly the same vocabulary and settings. The whole technique, including feature extraction, requires 22 ms/frame in a sequence with 26 300 images that is one order of magnitude faster than previous approaches.
