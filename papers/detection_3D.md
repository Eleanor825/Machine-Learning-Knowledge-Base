<span style="font-family:monospace">

# Papers in Computer Vision - 3D Object Detection

count: 5

* [Part-$A^{2}$ Net](https://arxiv.org/abs/1907.03670)
    * Title: From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    * Year: 08 Jul 2019
    * Author: Shaoshuai Shi
    * Abstract: 3D object detection from LiDAR point cloud is a challenging problem in 3D scene understanding and has many practical applications. In this paper, we extend our preliminary work PointRCNN to a novel and strong point-cloud-based 3D object detection framework, the part-aware and aggregation neural network (Part-$A^{2}$ net). The whole framework consists of the part-aware stage and the part-aggregation stage. Firstly, the part-aware stage for the first time fully utilizes free-of-charge part supervisions derived from 3D ground-truth boxes to simultaneously predict high quality 3D proposals and accurate intra-object part locations. The predicted intra-object part locations within the same proposal are grouped by our new-designed RoI-aware point cloud pooling module, which results in an effective representation to encode the geometry-specific features of each 3D proposal. Then the part-aggregation stage learns to re-score the box and refine the box location by exploring the spatial relationship of the pooled intra-object part locations. Extensive experiments are conducted to demonstrate the performance improvements from each component of our proposed framework. Our Part-$A^{2}$ net outperforms all existing 3D detection methods and achieves new state-of-the-art on KITTI 3D object detection dataset by utilizing only the LiDAR point cloud data. Code is available at [this https URL](https://github.com/sshaoshuai/PointCloudDet3D).
* [PointRCNN](https://arxiv.org/abs/1812.04244)
    * Title: PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    * Year: 11 Dec 2018
    * Author: Shaoshuai Shi
    * Abstract: In this paper, we propose PointRCNN for 3D object detection from raw point cloud. The whole framework is composed of two stages: stage-1 for the bottom-up 3D proposal generation and stage-2 for refining proposals in the canonical coordinates to obtain the final detection results. Instead of generating proposals from RGB image or projecting point cloud to bird's view or voxels as previous methods do, our stage-1 sub-network directly generates a small number of high-quality 3D proposals from point cloud in a bottom-up manner via segmenting the point cloud of the whole scene into foreground points and background. The stage-2 sub-network transforms the pooled points of each proposal to canonical coordinates to learn better local spatial features, which is combined with global semantic features of each point learned in stage-1 for accurate box refinement and confidence prediction. Extensive experiments on the 3D detection benchmark of KITTI dataset show that our proposed architecture outperforms state-of-the-art methods with remarkable margins by using only point cloud as input. The code is available at [this https URL](https://github.com/sshaoshuai/PointRCNN).
* [STD](https://arxiv.org/abs/1907.10471)
    * Title: STD: Sparse-to-Dense 3D Object Detector for Point Cloud
    * Year: 22 Jul 2019
    * Author: Zetong Yang
    * Abstract: We present a new two-stage 3D object detection framework, named sparse-to-dense 3D Object Detector (STD). The first stage is a bottom-up proposal generation network that uses raw point cloud as input to generate accurate proposals by seeding each point with a new spherical anchor. It achieves a high recall with less computation compared with prior works. Then, PointsPool is applied for generating proposal features by transforming their interior point features from sparse expression to compact representation, which saves even more computation time. In box prediction, which is the second stage, we implement a parallel intersection-over-union (IoU) branch to increase awareness of localization accuracy, resulting in further improved performance. We conduct experiments on KITTI dataset, and evaluate our method in terms of 3D object and Bird's Eye View (BEV) detection. Our method outperforms other state-of-the-arts by a large margin, especially on the hard set, with inference speed more than 10 FPS.
* [PV-RCNN++](https://arxiv.org/abs/2102.00463)
    * Title: PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection
    * Year: 31 Jan 2021
    * Author: Shaoshuai Shi
    * Abstract: 3D object detection is receiving increasing attention from both industry and academia thanks to its wide applications in various fields. In this paper, we propose the Point-Voxel Region-based Convolution Neural Networks (PV-RCNNs) for 3D object detection from point clouds. First, we propose a novel 3D detector, PV-RCNN, which consists of two steps: the voxel-to-keypoint scene encoding and keypoint-to-grid RoI feature abstraction. These two steps deeply integrate the 3D voxel CNN with the PointNet-based set abstraction for extracting discriminative features. Second, we propose an advanced framework, PV-RCNN++, for more efficient and accurate 3D object detection. It consists of two major improvements: the sectorized proposal-centric strategy for efficiently producing more representative keypoints, and the VectorPool aggregation for better aggregating local point features with much less resource consumption. With these two strategies, our PV-RCNN++ is more than 2x faster than PV-RCNN, while also achieving better performance on the large-scale Waymo Open Dataset with 150m * 150m detection range. Also, our proposed PV-RCNNs achieve state-of-the-art 3D detection performance on both the Waymo Open Dataset and the highly-competitive KITTI benchmark. The source code is available at [this https URL](https://github.com/open-mmlab/OpenPCDet).
* [PointPillars](https://arxiv.org/abs/1812.05784)
    * Title: PointPillars: Fast Encoders for Object Detection from Point Clouds
    * Year: 14 Dec 2018
    * Author: Alex H. Lang
    * Abstract: Object detection in point clouds is an important aspect of many robotics applications such as autonomous driving. In this paper we consider the problem of encoding a point cloud into a format appropriate for a downstream detection pipeline. Recent literature suggests two types of encoders; fixed encoders tend to be fast but sacrifice accuracy, while encoders that are learned from data are more accurate, but slower. In this work we propose PointPillars, a novel encoder which utilizes PointNets to learn a representation of point clouds organized in vertical columns (pillars). While the encoded features can be used with any standard 2D convolutional detection architecture, we further propose a lean downstream network. Extensive experimentation shows that PointPillars outperforms previous encoders with respect to both speed and accuracy by a large margin. Despite only using lidar, our full detection pipeline significantly outperforms the state of the art, even among fusion methods, with respect to both the 3D and bird's eye view KITTI benchmarks. This detection performance is achieved while running at 62 Hz: a 2 - 4 fold runtime improvement. A faster version of our method matches the state of the art at 105 Hz. These benchmarks suggest that PointPillars is an appropriate encoding for object detection in point clouds.