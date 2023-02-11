# [Papers][Vision] Pointcloud Processing <!-- omit in toc -->

count=2

## Table of Contents <!-- omit in toc -->

- [Detection](#detection)

----------------------------------------------------------------------------------------------------

## Detection

* [[VoxelNet](https://arxiv.org/abs/1711.06396)]
    [[pdf](https://arxiv.org/pdf/1711.06396.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1711.06396/)]
    * Title: VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
    * Year: 17 Nov `2017`
    * Authors: Yin Zhou, Oncel Tuzel
    * Abstract: Accurate detection of objects in 3D point clouds is a central problem in many applications, such as autonomous navigation, housekeeping robots, and augmented/virtual reality. To interface a highly sparse LiDAR point cloud with a region proposal network (RPN), most existing efforts have focused on hand-crafted feature representations, for example, a bird's eye view projection. In this work, we remove the need of manual feature engineering for 3D point clouds and propose VoxelNet, a generic 3D detection network that unifies feature extraction and bounding box prediction into a single stage, end-to-end trainable deep network. Specifically, VoxelNet divides a point cloud into equally spaced 3D voxels and transforms a group of points within each voxel into a unified feature representation through the newly introduced voxel feature encoding (VFE) layer. In this way, the point cloud is encoded as a descriptive volumetric representation, which is then connected to a RPN to generate detections. Experiments on the KITTI car detection benchmark show that VoxelNet outperforms the state-of-the-art LiDAR based 3D detection methods by a large margin. Furthermore, our network learns an effective discriminative representation of objects with various geometries, leading to encouraging results in 3D detection of pedestrians and cyclists, based on only LiDAR.
* [[PointPillars](https://arxiv.org/abs/1812.05784)]
    [[pdf](https://arxiv.org/pdf/1812.05784.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1812.05784/)]
    * Title: PointPillars: Fast Encoders for Object Detection from Point Clouds
    * Year: 14 Dec `2018`
    * Authors: Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom
    * Abstract: Object detection in point clouds is an important aspect of many robotics applications such as autonomous driving. In this paper we consider the problem of encoding a point cloud into a format appropriate for a downstream detection pipeline. Recent literature suggests two types of encoders; fixed encoders tend to be fast but sacrifice accuracy, while encoders that are learned from data are more accurate, but slower. In this work we propose PointPillars, a novel encoder which utilizes PointNets to learn a representation of point clouds organized in vertical columns (pillars). While the encoded features can be used with any standard 2D convolutional detection architecture, we further propose a lean downstream network. Extensive experimentation shows that PointPillars outperforms previous encoders with respect to both speed and accuracy by a large margin. Despite only using lidar, our full detection pipeline significantly outperforms the state of the art, even among fusion methods, with respect to both the 3D and bird's eye view KITTI benchmarks. This detection performance is achieved while running at 62 Hz: a 2 - 4 fold runtime improvement. A faster version of our method matches the state of the art at 105 Hz. These benchmarks suggest that PointPillars is an appropriate encoding for object detection in point clouds.
