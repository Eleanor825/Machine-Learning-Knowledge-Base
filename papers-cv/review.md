<span style="font-family:monospace">

# Review/Survey Papers

count: 6

## Others

* [Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538)
    * Title: Representation Learning: A Review and New Perspectives
    * Year: 24 Jun 2012
    * Author: Yoshua Bengio
    * Abstract: The success of machine learning algorithms generally depends on data representation, and we hypothesize that this is because different representations can entangle and hide more or less the different explanatory factors of variation behind the data. Although specific domain knowledge can be used to help design representations, learning with generic priors can also be used, and the quest for AI is motivating the design of more powerful representation-learning algorithms implementing such priors. This paper reviews recent work in the area of unsupervised feature learning and deep learning, covering advances in probabilistic models, auto-encoders, manifold learning, and deep networks. This motivates longer-term unanswered questions about the appropriate objectives for learning good representations, for computing representations (i.e., inference), and the geometrical connections between representation learning, density estimation and manifold learning.

## SLAM

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