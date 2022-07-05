<span style="font-family:monospace">

# Papers in Computer Vision - SLAM

count: 4

* [SVO](https://ieeexplore.ieee.org/document/6906584)
    * Title: SVO: Fast semi-direct monocular visual odometry
    * Year: `2014`
    * Author: Christian Forster
    * Abstract: We propose a semi-direct monocular visual odometry algorithm that is precise, robust, and faster than current state-of-the-art methods. The semi-direct approach eliminates the need of costly feature extraction and robust matching techniques for motion estimation. Our algorithm operates directly on pixel intensities, which results in subpixel precision at high frame-rates. A probabilistic mapping method that explicitly models outlier measurements is used to estimate 3D points, which results in fewer outliers and more reliable points. Precise and high frame-rate motion estimation brings increased robustness in scenes of little, repetitive, and high-frequency texture. The algorithm is applied to micro-aerial-vehicle state-estimation in GPS-denied environments and runs at 55 frames per second on the onboard embedded computer and at more than 300 frames per second on a consumer laptop. We call our approach SVO (Semi-direct Visual Odometry) and release our implementation as open-source software.
* [ORB-SLAM](https://arxiv.org/abs/1502.00956)
    * Title: ORB-SLAM: a Versatile and Accurate Monocular SLAM System
    * Year: 03 Feb `2015`
    * Author: Raul Mur-Artal
    * Abstract: This paper presents ORB-SLAM, a feature-based monocular SLAM system that operates in real time, in small and large, indoor and outdoor environments. The system is robust to severe motion clutter, allows wide baseline loop closing and relocalization, and includes full automatic initialization. Building on excellent algorithms of recent years, we designed from scratch a novel system that uses the same features for all SLAM tasks: tracking, mapping, relocalization, and loop closing. A survival of the fittest strategy that selects the points and keyframes of the reconstruction leads to excellent robustness and generates a compact and trackable map that only grows if the scene content changes, allowing lifelong operation. We present an exhaustive evaluation in 27 sequences from the most popular datasets. ORB-SLAM achieves unprecedented performance with respect to other state-of-the-art monocular SLAM approaches. For the benefit of the community, we make the source code public.
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
