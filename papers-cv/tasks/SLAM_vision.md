# [Papers][Vision][SLAM] Vision SLAM

count: 16

## Parallel Tracking and Mapping (PTAM)

* [Parallel Tracking and Mapping for Small AR Workspaces](https://ieeexplore.ieee.org/document/4538852)
    * Title: Parallel Tracking and Mapping for Small AR Workspaces
    * Year: `2007`
    * Author: Georg Klein
    * Abstract: This paper presents a method of estimating camera pose in an unknown scene. While this has previously been attempted by adapting SLAM algorithms developed for robotic exploration, we propose a system specifically designed to track a hand-held camera in a small AR workspace. We propose to split tracking and mapping into two separate tasks, processed in parallel threads on a dual-core computer: one thread deals with the task of robustly tracking erratic hand-held motion, while the other produces a 3D map of point features from previously observed video frames. This allows the use of computationally expensive batch optimisation techniques not usually associated with real-time operation: The result is a system that produces detailed maps with thousands of landmarks which can be tracked at frame-rate, with an accuracy and robustness rivalling that of state-of-the-art model-based systems.
* [Improving the Agility of Keyframe-Based SLAM](https://dl.acm.org/doi/10.1007/978-3-540-88688-4_59)
    * Title: Improving the Agility of Keyframe-Based SLAM
    * Year: 
    * Authors: 
    * Abstract: 
* [Parallel Tracking and Mapping on a camera phone](https://ieeexplore.ieee.org/document/5336495)
    * Title: Parallel Tracking and Mapping on a camera phone
    * Year: 
    * Authors: 
    * Abstract: 

## Large-Scale Direct SLAM (LSD-SLAM)

* [LSD-SLAM: Large-scale direct monocular SLAM]
* [Large-scale direct SLAM with stereo cameras](https://ieeexplore.ieee.org/document/7353631)
    * Title: Large-scale direct SLAM with stereo cameras
    * Year: 17 December `2015`
    * Authors: Jakob Engel; Jörg Stückler; Daniel Cremers
    * Abstract: We propose a novel Large-Scale Direct SLAM algorithm for stereo cameras (Stereo LSD-SLAM) that runs in real-time at high frame rate on standard CPUs. In contrast to sparse interest-point based methods, our approach aligns images directly based on the photoconsistency of all high-contrast pixels, including corners, edges and high texture areas. It concurrently estimates the depth at these pixels from two types of stereo cues: Static stereo through the fixed-baseline stereo camera setup as well as temporal multi-view stereo exploiting the camera motion. By incorporating both disparity sources, our algorithm can even estimate depth of pixels that are under-constrained when only using fixed-baseline stereo. Using a fixed baseline, on the other hand, avoids scale-drift that typically occurs in pure monocular SLAM.We furthermore propose a robust approach to enforce illumination invariance, capable of handling aggressive brightness changes between frames - greatly improving the performance in realistic settings. In experiments, we demonstrate state-of-the-art results on stereo SLAM benchmarks such as Kitti or challenging datasets from the EuRoC Challenge 3 for micro aerial vehicles.

## SVO

* [SVO](https://ieeexplore.ieee.org/document/6906584)
    * Title: SVO: Fast semi-direct monocular visual odometry
    * Year: `2014`
    * Author: Christian Forster
    * Abstract: We propose a semi-direct monocular visual odometry algorithm that is precise, robust, and faster than current state-of-the-art methods. The semi-direct approach eliminates the need of costly feature extraction and robust matching techniques for motion estimation. Our algorithm operates directly on pixel intensities, which results in subpixel precision at high frame-rates. A probabilistic mapping method that explicitly models outlier measurements is used to estimate 3D points, which results in fewer outliers and more reliable points. Precise and high frame-rate motion estimation brings increased robustness in scenes of little, repetitive, and high-frequency texture. The algorithm is applied to micro-aerial-vehicle state-estimation in GPS-denied environments and runs at 55 frames per second on the onboard embedded computer and at more than 300 frames per second on a consumer laptop. We call our approach SVO (Semi-direct Visual Odometry) and release our implementation as open-source software.
* [SVO: Semidirect Visual Odometry for Monocular and Multicamera Systems](https://ieeexplore.ieee.org/document/7782863)
    * Title: SVO: Semidirect Visual Odometry for Monocular and Multicamera Systems
    * Year: 
    * Authors: 
    * Abstract: 

## Oriented FAST and Rotated BRIEF (ORB)

* [ORB](https://ieeexplore.ieee.org/document/6126544)
    * Title: ORB: An efficient alternative to SIFT or SURF
    * Year: 12 January `2012`
    * Authors: Ethan Rublee; Vincent Rabaud; Kurt Konolige; Gary Bradski
    * Abstract: Feature matching is at the base of many computer vision problems, such as object recognition or structure from motion. Current methods rely on costly descriptors for detection and matching. In this paper, we propose a very fast binary descriptor based on BRIEF, called ORB, which is rotation invariant and resistant to noise. We demonstrate through experiments how ORB is at two orders of magnitude faster than SIFT, while performing as well in many situations. The efficiency is tested on several real-world applications, including object detection and patch-tracking on a smart phone.
* [ORB-SLAM1](https://arxiv.org/abs/1502.00956)
    * Title: ORB-SLAM: a Versatile and Accurate Monocular SLAM System
    * Year: 03 Feb `2015`
    * Authors: Raul Mur-Artal, J. M. M. Montiel, Juan D. Tardos
    * Abstract: This paper presents ORB-SLAM, a feature-based monocular SLAM system that operates in real time, in small and large, indoor and outdoor environments. The system is robust to severe motion clutter, allows wide baseline loop closing and relocalization, and includes full automatic initialization. Building on excellent algorithms of recent years, we designed from scratch a novel system that uses the same features for all SLAM tasks: tracking, mapping, relocalization, and loop closing. A survival of the fittest strategy that selects the points and keyframes of the reconstruction leads to excellent robustness and generates a compact and trackable map that only grows if the scene content changes, allowing lifelong operation. We present an exhaustive evaluation in 27 sequences from the most popular datasets. ORB-SLAM achieves unprecedented performance with respect to other state-of-the-art monocular SLAM approaches. For the benefit of the community, we make the source code public.
* [ORB-SLAM2](https://arxiv.org/abs/1610.06475)
    * Title: ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras
    * Year: 20 Oct `2016`
    * Authors: Raul Mur-Artal, Juan D. Tardos
    * Abstract: We present ORB-SLAM2 a complete SLAM system for monocular, stereo and RGB-D cameras, including map reuse, loop closing and relocalization capabilities. The system works in real-time on standard CPUs in a wide variety of environments from small hand-held indoors sequences, to drones flying in industrial environments and cars driving around a city. Our back-end based on bundle adjustment with monocular and stereo observations allows for accurate trajectory estimation with metric scale. Our system includes a lightweight localization mode that leverages visual odometry tracks for unmapped regions and matches to map points that allow for zero-drift localization. The evaluation on 29 popular public sequences shows that our method achieves state-of-the-art accuracy, being in most cases the most accurate SLAM solution. We publish the source code, not only for the benefit of the SLAM community, but with the aim of being an out-of-the-box SLAM solution for researchers in other fields.
* [ORB-SLAM3](https://arxiv.org/abs/2007.11898)
    * Title: ORB-SLAM3: An accurate open-source library for visual, visual-inertial and multi-map SLAM
    * Year: 23 Jul `2020`
    * Authors: Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, Juan D. Tardós
    * Abstract: This paper presents ORB-SLAM3, the first system able to perform visual, visual-inertial and multi-map SLAM with monocular, stereo and RGB-D cameras, using pin-hole and fisheye lens models. The first main novelty is a feature-based tightly-integrated visual-inertial SLAM system that fully relies on Maximum-a-Posteriori (MAP) estimation, even during the IMU initialization phase. The result is a system that operates robustly in real-time, in small and large, indoor and outdoor environments, and is 2 to 5 times more accurate than previous approaches. The second main novelty is a multiple map system that relies on a new place recognition method with improved recall. Thanks to it, ORB-SLAM3 is able to survive to long periods of poor visual information: when it gets lost, it starts a new map that will be seamlessly merged with previous maps when revisiting mapped areas. Compared with visual odometry systems that only use information from the last few seconds, ORB-SLAM3 is the first system able to reuse in all the algorithm stages all previous information. This allows to include in bundle adjustment co-visible keyframes, that provide high parallax observations boosting accuracy, even if they are widely separated in time or if they come from a previous mapping session. Our experiments show that, in all sensor configurations, ORB-SLAM3 is as robust as the best systems available in the literature, and significantly more accurate. Notably, our stereo-inertial SLAM achieves an average accuracy of 3.6 cm on the EuRoC drone and 9 mm under quick hand-held motions in the room of TUM-VI dataset, a setting representative of AR/VR scenarios. For the benefit of the community we make public the source code.
* [ORBSLAM-Atlas](https://arxiv.org/abs/1908.11585)
    * Title: ORBSLAM-Atlas: a robust and accurate multi-map system
    * Year: 30 Aug `2019`
    * Authors: Richard Elvira, Juan D. Tardós, J.M.M. Montiel
    * Abstract: We propose ORBSLAM-Atlas, a system able to handle an unlimited number of disconnected sub-maps, that includes a robust map merging algorithm able to detect sub-maps with common regions and seamlessly fuse them. The outstanding robustness and accuracy of ORBSLAM are due to its ability to detect wide-baseline matches between keyframes, and to exploit them by means of non-linear optimization, however it only can handle a single map. ORBSLAM-Atlas brings the wide-baseline matching detection and exploitation to the multiple map arena. The result is a SLAM system significantly more general and robust, able to perform multi-session mapping. If tracking is lost during exploration, instead of freezing the map, a new sub-map is launched, and it can be fused with the previous map when common parts are visited. Our criteria to declare the camera lost contrast with previous approaches that simply count the number of tracked points, we propose to discard also inaccurately estimated camera poses due to bad geometrical conditioning. As a result, the map is split into more accurate sub-maps, that are eventually merged in a more accurate global map, thanks to the multi-mapping capabilities. We provide extensive experimental validation in the EuRoC datasets, where ORBSLAM-Atlas obtains accurate monocular and stereo results in the difficult sequences where ORBSLAM failed. We also build global maps after multiple sessions in the same room, obtaining the best results to date, between 2 and 3 times more accurate than competing multi-map approaches. We also show the robustness and capability of our system to deal with dynamic scenes, quantitatively in the EuRoC datasets and qualitatively in a densely populated corridor where camera occlusions and tracking losses are frequent.
* [Inertial-Only Optimization for Visual-Inertial Initialization](https://arxiv.org/abs/2003.05766)
    * Title: Inertial-Only Optimization for Visual-Inertial Initialization
    * Year: 12 Mar `2020`
    * Authors: Carlos Campos, José M.M. Montiel, Juan D. Tardós
    * Abstract: We formulate for the first time visual-inertial initialization as an optimal estimation problem, in the sense of maximum-a-posteriori (MAP) estimation. This allows us to properly take into account IMU measurement uncertainty, which was neglected in previous methods that either solved sets of algebraic equations, or minimized ad-hoc cost functions using least squares. Our exhaustive initialization tests on EuRoC dataset show that our proposal largely outperforms the best methods in the literature, being able to initialize in less than 4 seconds in almost any point of the trajectory, with a scale error of 5.3% on average. This initialization has been integrated into ORB-SLAM Visual-Inertial boosting its robustness and efficiency while maintaining its excellent accuracy.

## Direct Sparse Odometry (DSO)

* [DSO](https://arxiv.org/abs/1607.02565)
    * Title: Direct Sparse Odometry
    * Year: 09 Jul `2016`
    * Author: Jakob Engel
    * Abstract: We propose a novel direct sparse visual odometry formulation. It combines a fully direct probabilistic model (minimizing a photometric error) with consistent, joint optimization of all model parameters, including geometry -- represented as inverse depth in a reference frame -- and camera motion. This is achieved in real time by omitting the smoothness prior used in other direct methods and instead sampling pixels evenly throughout the images. Since our method does not depend on keypoint detectors or descriptors, it can naturally sample pixels from across all image regions that have intensity gradient, including edges or smooth intensity variations on mostly white walls. The proposed model integrates a full photometric calibration, accounting for exposure time, lens vignetting, and non-linear response functions. We thoroughly evaluate our method on three different datasets comprising several hours of video. The experiments show that the presented approach significantly outperforms state-of-the-art direct and indirect methods in a variety of real-world settings, both in terms of tracking accuracy and robustness.
* [Omnidirectional DSO](https://arxiv.org/abs/1808.02775)
    * Title: Omnidirectional DSO: Direct Sparse Odometry with Fisheye Cameras
    * Year: 08 Aug `2018`
    * Authors: Hidenobu Matsuki, Lukas von Stumberg, Vladyslav Usenko, Jörg Stückler, Daniel Cremers
    * Abstract: We propose a novel real-time direct monocular visual odometry for omnidirectional cameras. Our method extends direct sparse odometry (DSO) by using the unified omnidirectional model as a projection function, which can be applied to fisheye cameras with a field-of-view (FoV) well above 180 degrees. This formulation allows for using the full area of the input image even with strong distortion, while most existing visual odometry methods can only use a rectified and cropped part of it. Model parameters within an active keyframe window are jointly optimized, including the intrinsic/extrinsic camera parameters, 3D position of points, and affine brightness parameters. Thanks to the wide FoV, image overlap between frames becomes bigger and points are more spatially distributed. Our results demonstrate that our method provides increased accuracy and robustness over state-of-the-art visual odometry algorithms.
* [Stereo DSO](https://arxiv.org/abs/1708.07878)
    * Title: Stereo DSO: Large-Scale Direct Sparse Visual Odometry with Stereo Cameras
    * Year: 25 Aug `2017`
    * Authors: Rui Wang, Martin Schwörer, Daniel Cremers
    * Abstract: We propose Stereo Direct Sparse Odometry (Stereo DSO) as a novel method for highly accurate real-time visual odometry estimation of large-scale environments from stereo cameras. It jointly optimizes for all the model parameters within the active window, including the intrinsic/extrinsic camera parameters of all keyframes and the depth values of all selected pixels. In particular, we propose a novel approach to integrate constraints from static stereo into the bundle adjustment pipeline of temporal multi-view stereo. Real-time optimization is realized by sampling pixels uniformly from image regions with sufficient intensity gradient. Fixed-baseline stereo resolves scale drift. It also reduces the sensitivities to large optical flow and to rolling shutter effect which are known shortcomings of direct image alignment methods. Quantitative evaluation demonstrates that the proposed Stereo DSO outperforms existing state-of-the-art visual odometry methods both in terms of tracking accuracy and robustness. Moreover, our method delivers a more precise metric 3D reconstruction than previous dense/semi-dense direct approaches while providing a higher reconstruction density than feature-based methods.
