# [Papers][Vision] NERF <!-- omit in toc -->

count=1

## Table of Contents <!-- omit in toc -->

----------------------------------------------------------------------------------------------------

## Others

### Fri Feb 03, 2023 NERF Readings

* [[Point-NeRF: Point-based Neural Radiance Fields](https://arxiv.org/abs/2201.08845)]
    [[pdf](https://arxiv.org/pdf/2201.08845.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2201.08845/)]
    * Title: Point-NeRF: Point-based Neural Radiance Fields
    * Year: 21 Jan `2022`
    * Authors: Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, Ulrich Neumann
    * Abstract: Volumetric neural rendering methods like NeRF generate high-quality view synthesis results but are optimized per-scene leading to prohibitive reconstruction time. On the other hand, deep multi-view stereo methods can quickly reconstruct scene geometry via direct network inference. Point-NeRF combines the advantages of these two approaches by using neural 3D point clouds, with associated neural features, to model a radiance field. Point-NeRF can be rendered efficiently by aggregating neural point features near scene surfaces, in a ray marching-based rendering pipeline. Moreover, Point-NeRF can be initialized via direct inference of a pre-trained deep network to produce a neural point cloud; this point cloud can be finetuned to surpass the visual quality of NeRF with 30X faster training time. Point-NeRF can be combined with other 3D reconstruction methods and handles the errors and outliers in such methods via a novel pruning and growing mechanism. The experiments on the DTU, the NeRF Synthetics , the ScanNet and the Tanks and Temples datasets demonstrate Point-NeRF can surpass the existing methods and achieve the state-of-the-art results.
