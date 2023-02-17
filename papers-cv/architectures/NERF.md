# [Papers][Vision] NERF <!-- omit in toc -->

count=11

## Table of Contents <!-- omit in toc -->

- [1. ChatGPT Recommendations](#1-chatgpt-recommendations)
- [2. Readings](#2-readings)
  - [2.1. Fri Feb 03, 2023 NERF Readings](#21-fri-feb-03-2023-nerf-readings)
  - [2.2. Fri Feb 17, 2023 NERF Readings](#22-fri-feb-17-2023-nerf-readings)

----------------------------------------------------------------------------------------------------

## 1. ChatGPT Recommendations

* [[NeRF--](https://arxiv.org/abs/2102.07064)]
    [[pdf](https://arxiv.org/pdf/2102.07064.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2102.07064/)]
    * Title: NeRF--: Neural Radiance Fields Without Known Camera Parameters
    * Year: 14 Feb `2021`
    * Authors: Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, Victor Adrian Prisacariu
    * Abstract: Considering the problem of novel view synthesis (NVS) from only a set of 2D images, we simplify the training process of Neural Radiance Field (NeRF) on forward-facing scenes by removing the requirement of known or pre-computed camera parameters, including both intrinsics and 6DoF poses. To this end, we propose NeRF$--$, with three contributions: First, we show that the camera parameters can be jointly optimised as learnable parameters with NeRF training, through a photometric reconstruction; Second, to benchmark the camera parameter estimation and the quality of novel view renderings, we introduce a new dataset of path-traced synthetic scenes, termed as Blender Forward-Facing Dataset (BLEFF); Third, we conduct extensive analyses to understand the training behaviours under various camera motions, and show that in most scenarios, the joint optimisation pipeline can recover accurate camera parameters and achieve comparable novel view synthesis quality as those trained with COLMAP pre-computed camera parameters. Our code and data are available at https://nerfmm.active.vision.
* [[DeepSDF](https://arxiv.org/abs/1901.05103)]
    [[pdf](https://arxiv.org/pdf/1901.05103.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1901.05103/)]
    * Title: DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation
    * Year: 16 Jan `2019`
    * Authors: Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, Steven Lovegrove
    * Abstract: Computer graphics, 3D computer vision and robotics communities have produced multiple approaches to representing 3D geometry for rendering and reconstruction. These provide trade-offs across fidelity, efficiency and compression capabilities. In this work, we introduce DeepSDF, a learned continuous Signed Distance Function (SDF) representation of a class of shapes that enables high quality shape representation, interpolation and completion from partial and noisy 3D input data. DeepSDF, like its classical counterpart, represents a shape's surface by a continuous volumetric field: the magnitude of a point in the field represents the distance to the surface boundary and the sign indicates whether the region is inside (-) or outside (+) of the shape, hence our representation implicitly encodes a shape's boundary as the zero-level-set of the learned function while explicitly representing the classification of space as being part of the shapes interior or not. While classical SDF's both in analytical or discretized voxel form typically represent the surface of a single shape, DeepSDF can represent an entire class of shapes. Furthermore, we show state-of-the-art performance for learned 3D shape representation and completion while reducing the model size by an order of magnitude compared with previous work.
* [[AtlasNet](https://arxiv.org/abs/1802.05384)]
    [[pdf](https://arxiv.org/pdf/1802.05384.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1802.05384/)]
    * Title: AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation
    * Year: 15 Feb `2018`
    * Authors: Thibault Groueix, Matthew Fisher, Vladimir G. Kim, Bryan C. Russell, Mathieu Aubry
    * Abstract: We introduce a method for learning to generate the surface of 3D shapes. Our approach represents a 3D shape as a collection of parametric surface elements and, in contrast to methods generating voxel grids or point clouds, naturally infers a surface representation of the shape. Beyond its novelty, our new shape generation framework, AtlasNet, comes with significant advantages, such as improved precision and generalization capabilities, and the possibility to generate a shape of arbitrary resolution without memory issues. We demonstrate these benefits and compare to strong baselines on the ShapeNet benchmark for two applications: (i) auto-encoding shapes, and (ii) single-view reconstruction from a still image. We also provide results showing its potential for other applications, such as morphing, parametrization, super-resolution, matching, and co-segmentation.
* [[Neural Volumes](https://arxiv.org/abs/1906.07751)]
    [[pdf](https://arxiv.org/pdf/1906.07751.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1906.07751/)]
    * Title: Neural Volumes: Learning Dynamic Renderable Volumes from Images
    * Year: 18 Jun `2019`
    * Authors: Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, Yaser Sheikh
    * Abstract: Modeling and rendering of dynamic scenes is challenging, as natural scenes often contain complex phenomena such as thin structures, evolving topology, translucency, scattering, occlusion, and biological motion. Mesh-based reconstruction and tracking often fail in these cases, and other approaches (e.g., light field video) typically rely on constrained viewing conditions, which limit interactivity. We circumvent these difficulties by presenting a learning-based approach to representing dynamic objects inspired by the integral projection model used in tomographic imaging. The approach is supervised directly from 2D images in a multi-view capture setting and does not require explicit reconstruction or tracking of the object. Our method has two primary components: an encoder-decoder network that transforms input images into a 3D volume representation, and a differentiable ray-marching operation that enables end-to-end training. By virtue of its 3D representation, our construction extrapolates better to novel viewpoints compared to screen-space rendering techniques. The encoder-decoder architecture learns a latent representation of a dynamic scene that enables us to produce novel content sequences not seen during training. To overcome memory limitations of voxel-based representations, we learn a dynamic irregular grid structure implemented with a warp field during ray-marching. This structure greatly improves the apparent resolution and reduces grid-like artifacts and jagged motion. Finally, we demonstrate how to incorporate surface-based representations into our volumetric-learning framework for applications where the highest resolution is required, using facial performance capture as a case in point.
* [[NeRF](https://arxiv.org/abs/2003.08934)]
    [[pdf](https://arxiv.org/pdf/2003.08934.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2003.08934/)]
    * Title: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
    * Year: 19 Mar `2020`
    * Authors: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
    * Abstract: We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully-connected (non-convolutional) deep network, whose input is a single continuous 5D coordinate (spatial location $(x,y,z)$ and viewing direction $(\theta, \phi)$) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis. View synthesis results are best viewed as videos, so we urge readers to view our supplementary video for convincing comparisons.
* [[GRAF](https://arxiv.org/abs/2007.02442)]
    [[pdf](https://arxiv.org/pdf/2007.02442.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2007.02442/)]
    * Title: GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis
    * Year: 05 Jul `2020`
    * Authors: Katja Schwarz, Yiyi Liao, Michael Niemeyer, Andreas Geiger
    * Abstract: While 2D generative adversarial networks have enabled high-resolution image synthesis, they largely lack an understanding of the 3D world and the image formation process. Thus, they do not provide precise control over camera viewpoint or object pose. To address this problem, several recent approaches leverage intermediate voxel-based representations in combination with differentiable rendering. However, existing methods either produce low image resolution or fall short in disentangling camera and scene properties, e.g., the object identity may vary with the viewpoint. In this paper, we propose a generative model for radiance fields which have recently proven successful for novel view synthesis of a single scene. In contrast to voxel-based representations, radiance fields are not confined to a coarse discretization of the 3D space, yet allow for disentangling camera and scene properties while degrading gracefully in the presence of reconstruction ambiguity. By introducing a multi-scale patch-based discriminator, we demonstrate synthesis of high-resolution images while training our model from unposed 2D images alone. We systematically analyze our approach on several challenging synthetic and real-world datasets. Our experiments reveal that radiance fields are a powerful representation for generative image synthesis, leading to 3D consistent models that render with high fidelity.
* [[NeRF in the Wild](https://arxiv.org/abs/2008.02268)]
    [[pdf](https://arxiv.org/pdf/2008.02268.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2008.02268/)]
    * Title: NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections
    * Year: 05 Aug `2020`
    * Authors: Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, Daniel Duckworth
    * Abstract: We present a learning-based method for synthesizing novel views of complex scenes using only unstructured collections of in-the-wild photographs. We build on Neural Radiance Fields (NeRF), which uses the weights of a multilayer perceptron to model the density and color of a scene as a function of 3D coordinates. While NeRF works well on images of static subjects captured under controlled settings, it is incapable of modeling many ubiquitous, real-world phenomena in uncontrolled images, such as variable illumination or transient occluders. We introduce a series of extensions to NeRF to address these issues, thereby enabling accurate reconstructions from unstructured image collections taken from the internet. We apply our system, dubbed NeRF-W, to internet photo collections of famous landmarks, and demonstrate temporally consistent novel view renderings that are significantly closer to photorealism than the prior state of the art.
* [[NeX](https://arxiv.org/abs/2103.05606)]
    [[pdf](https://arxiv.org/pdf/2103.05606.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2103.05606/)]
    * Title: NeX: Real-time View Synthesis with Neural Basis Expansion
    * Year: 09 Mar `2021`
    * Authors: Suttisak Wizadwongsa, Pakkapon Phongthawee, Jiraphon Yenphraphai, Supasorn Suwajanakorn
    * Abstract: We present NeX, a new approach to novel view synthesis based on enhancements of multiplane image (MPI) that can reproduce next-level view-dependent effects -- in real time. Unlike traditional MPI that uses a set of simple RGB$\alpha$ planes, our technique models view-dependent effects by instead parameterizing each pixel as a linear combination of basis functions learned from a neural network. Moreover, we propose a hybrid implicit-explicit modeling strategy that improves upon fine detail and produces state-of-the-art results. Our method is evaluated on benchmark forward-facing datasets as well as our newly-introduced dataset designed to test the limit of view-dependent modeling with significantly more challenging effects such as rainbow reflections on a CD. Our method achieves the best overall scores across all major metrics on these datasets with more than 1000$\times$ faster rendering time than the state of the art. For real-time demos, visit this https URL
* [[Neural Sparse Voxel Fields](https://arxiv.org/abs/2007.11571)]
    [[pdf](https://arxiv.org/pdf/2007.11571.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2007.11571/)]
    * Title: Neural Sparse Voxel Fields
    * Year: 22 Jul `2020`
    * Authors: Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, Christian Theobalt
    * Abstract: Photo-realistic free-viewpoint rendering of real-world scenes using classical computer graphics techniques is challenging, because it requires the difficult step of capturing detailed appearance and geometry models. Recent studies have demonstrated promising results by learning scene representations that implicitly encode both geometry and appearance without 3D supervision. However, existing approaches in practice often show blurry renderings caused by the limited network capacity or the difficulty in finding accurate intersections of camera rays with the scene geometry. Synthesizing high-resolution imagery from these representations often requires time-consuming optical ray marching. In this work, we introduce Neural Sparse Voxel Fields (NSVF), a new neural scene representation for fast and high-quality free-viewpoint rendering. NSVF defines a set of voxel-bounded implicit fields organized in a sparse voxel octree to model local properties in each cell. We progressively learn the underlying voxel structures with a differentiable ray-marching operation from only a set of posed RGB images. With the sparse voxel octree structure, rendering novel views can be accelerated by skipping the voxels containing no relevant scene content. Our method is typically over 10 times faster than the state-of-the-art (namely, NeRF(Mildenhall et al., 2020)) at inference time while achieving higher quality results. Furthermore, by utilizing an explicit sparse voxel representation, our method can easily be applied to scene editing and scene composition. We also demonstrate several challenging tasks, including multi-scene learning, free-viewpoint rendering of a moving human, and large-scale scene rendering. Code and data are available at our website: this https URL.

## 2. Readings

### 2.1. Fri Feb 03, 2023 NERF Readings

* [[Point-NeRF](https://arxiv.org/abs/2201.08845)]
    [[pdf](https://arxiv.org/pdf/2201.08845.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2201.08845/)]
    * Title: Point-NeRF: Point-based Neural Radiance Fields
    * Year: 21 Jan `2022`
    * Authors: Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, Ulrich Neumann
    * Abstract: Volumetric neural rendering methods like NeRF generate high-quality view synthesis results but are optimized per-scene leading to prohibitive reconstruction time. On the other hand, deep multi-view stereo methods can quickly reconstruct scene geometry via direct network inference. Point-NeRF combines the advantages of these two approaches by using neural 3D point clouds, with associated neural features, to model a radiance field. Point-NeRF can be rendered efficiently by aggregating neural point features near scene surfaces, in a ray marching-based rendering pipeline. Moreover, Point-NeRF can be initialized via direct inference of a pre-trained deep network to produce a neural point cloud; this point cloud can be finetuned to surpass the visual quality of NeRF with 30X faster training time. Point-NeRF can be combined with other 3D reconstruction methods and handles the errors and outliers in such methods via a novel pruning and growing mechanism. The experiments on the DTU, the NeRF Synthetics , the ScanNet and the Tanks and Temples datasets demonstrate Point-NeRF can surpass the existing methods and achieve the state-of-the-art results.

### 2.2. Fri Feb 17, 2023 NERF Readings

* [[Plenoxels](https://arxiv.org/abs/2112.05131)]
    [[pdf](https://arxiv.org/pdf/2112.05131.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2112.05131/)]
    * Title: Plenoxels: Radiance Fields without Neural Networks
    * Year: 09 Dec `2021`
    * Authors: Alex Yu, Sara Fridovich-Keil, Matthew Tancik, Qinhong Chen, Benjamin Recht, Angjoo Kanazawa
    * Abstract: We introduce Plenoxels (plenoptic voxels), a system for photorealistic view synthesis. Plenoxels represent a scene as a sparse 3D grid with spherical harmonics. This representation can be optimized from calibrated images via gradient methods and regularization without any neural components. On standard, benchmark tasks, Plenoxels are optimized two orders of magnitude faster than Neural Radiance Fields with no loss in visual quality.
