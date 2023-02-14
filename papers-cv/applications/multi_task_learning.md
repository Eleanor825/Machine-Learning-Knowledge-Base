# [Papers] Multi-Task Learning <!-- omit in toc -->

count=7

## Table of Contents <!-- omit in toc -->

- [1. Survey](#1-survey)
- [2. Unclassified](#2-unclassified)

----------------------------------------------------------------------------------------------------

## 1. Survey

* [[An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098)]
    [[pdf](https://arxiv.org/pdf/1706.05098.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1706.05098/)]
    * Title: An Overview of Multi-Task Learning in Deep Neural Networks
    * Year: 15 Jun `2017`
    * Authors: Sebastian Ruder
    * Abstract: Multi-task learning (MTL) has led to successes in many applications of machine learning, from natural language processing and speech recognition to computer vision and drug discovery. This article aims to give a general overview of MTL, particularly in deep neural networks. It introduces the two most common methods for MTL in Deep Learning, gives an overview of the literature, and discusses recent advances. In particular, it seeks to help ML practitioners apply MTL by shedding light on how MTL works and providing guidelines for choosing appropriate auxiliary tasks.

## 2. Unclassified

* [[PCGrad](https://arxiv.org/abs/2001.06782)]
    [[pdf](https://arxiv.org/pdf/2001.06782.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2001.06782/)]
    * Title: Gradient Surgery for Multi-Task Learning
    * Year: 19 Jan `2020`
    * Authors: Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn
    * Abstract: While deep learning and deep reinforcement learning (RL) systems have demonstrated impressive results in domains such as image classification, game playing, and robotic control, data efficiency remains a major challenge. Multi-task learning has emerged as a promising approach for sharing structure across multiple tasks to enable more efficient learning. However, the multi-task setting presents a number of optimization challenges, making it difficult to realize large efficiency gains compared to learning tasks independently. The reasons why multi-task learning is so challenging compared to single-task learning are not fully understood. In this work, we identify a set of three conditions of the multi-task optimization landscape that cause detrimental gradient interference, and develop a simple yet general approach for avoiding such interference between task gradients. We propose a form of gradient surgery that projects a task's gradient onto the normal plane of the gradient of any other task that has a conflicting gradient. On a series of challenging multi-task supervised and multi-task RL problems, this approach leads to substantial gains in efficiency and performance. Further, it is model-agnostic and can be combined with previously-proposed multi-task architectures for enhanced performance.
* [[Multi-task Deep Reinforcement Learning with PopArt](https://arxiv.org/abs/1809.04474)]
    [[pdf](https://arxiv.org/pdf/1809.04474.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1809.04474/)]
    * Title: Multi-task Deep Reinforcement Learning with PopArt
    * Year: 12 Sep `2018`
    * Authors: Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt, Hado van Hasselt
    * Abstract: The reinforcement learning community has made great strides in designing algorithms capable of exceeding human performance on specific tasks. These algorithms are mostly trained one task at the time, each new task requiring to train a brand new agent instance. This means the learning algorithm is general, but each solution is not; each agent can only solve the one task it was trained on. In this work, we study the problem of learning to master not one but multiple sequential-decision tasks at once. A general issue in multi-task learning is that a balance must be found between the needs of multiple tasks competing for the limited resources of a single learning system. Many learning algorithms can get distracted by certain tasks in the set of tasks to solve. Such tasks appear more salient to the learning process, for instance because of the density or magnitude of the in-task rewards. This causes the algorithm to focus on those salient tasks at the expense of generality. We propose to automatically adapt the contribution of each task to the agent's updates, so that all tasks have a similar impact on the learning dynamics. This resulted in state of the art performance on learning to play all games in a set of 57 diverse Atari games. Excitingly, our method learned a single trained policy - with a single set of weights - that exceeds median human performance. To our knowledge, this was the first time a single agent surpassed human-level performance on this multi-task domain. The same approach also demonstrated state of the art performance on a set of 30 tasks in the 3D reinforcement learning platform DeepMind Lab.
* [[GradNorm](https://arxiv.org/abs/1711.02257)]
    [[pdf](https://arxiv.org/pdf/1711.02257.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1711.02257/)]
    * Title: GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
    * Year: 07 Nov `2017`
    * Authors: Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich
    * Abstract: Deep multitask networks, in which one neural network produces multiple predictive outputs, can offer better speed and performance than their single-task counterparts but are challenging to train properly. We present a gradient normalization (GradNorm) algorithm that automatically balances training in deep multitask models by dynamically tuning gradient magnitudes. We show that for various network architectures, for both regression and classification tasks, and on both synthetic and real datasets, GradNorm improves accuracy and reduces overfitting across multiple tasks when compared to single-task networks, static baselines, and other adaptive multitask loss balancing techniques. GradNorm also matches or surpasses the performance of exhaustive grid search methods, despite only involving a single asymmetry hyperparameter $\alpha$. Thus, what was once a tedious search process that incurred exponentially more compute for each task added can now be accomplished within a few training runs, irrespective of the number of tasks. Ultimately, we will demonstrate that gradient manipulation affords us great control over the training dynamics of multitask networks and may be one of the keys to unlocking the potential of multitask learning.
* [[Ray Interference: a Source of Plateaus in Deep Reinforcement Learning](https://arxiv.org/abs/1904.11455)]
    [[pdf](https://arxiv.org/pdf/1904.11455.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1904.11455/)]
    * Title: Ray Interference: a Source of Plateaus in Deep Reinforcement Learning
    * Year: 25 Apr `2019`
    * Authors: Tom Schaul, Diana Borsa, Joseph Modayil, Razvan Pascanu
    * Abstract: Rather than proposing a new method, this paper investigates an issue present in existing learning algorithms. We study the learning dynamics of reinforcement learning (RL), specifically a characteristic coupling between learning and data generation that arises because RL agents control their future data distribution. In the presence of function approximation, this coupling can lead to a problematic type of 'ray interference', characterized by learning dynamics that sequentially traverse a number of performance plateaus, effectively constraining the agent to learn one thing at a time even when learning in parallel is better. We establish the conditions under which ray interference occurs, show its relation to saddle points and obtain the exact learning dynamics in a restricted setting. We characterize a number of its properties and discuss possible remedies.
* [[Cross-Stitch](https://arxiv.org/abs/1604.03539)]
    [[pdf](https://arxiv.org/pdf/1604.03539.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1604.03539/)]
    * Title: Cross-stitch Networks for Multi-task Learning
    * Year: 12 Apr `2016`
    * Authors: Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert
    * Abstract: Multi-task learning in Convolutional Networks has displayed remarkable success in the field of recognition. This success can be largely attributed to learning shared representations from multiple supervisory tasks. However, existing multi-task approaches rely on enumerating multiple network architectures specific to the tasks at hand, that do not generalize. In this paper, we propose a principled approach to learn shared representations in ConvNets using multi-task learning. Specifically, we propose a new sharing unit: "cross-stitch" unit. These units combine the activations from multiple networks and can be trained end-to-end. A network with cross-stitch units can learn an optimal combination of shared and task-specific representations. Our proposed method generalizes across multiple tasks and shows dramatically improved performance over baseline methods for categories with few training examples.
* [[End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704)]
    [[pdf](https://arxiv.org/pdf/1803.10704.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1803.10704/)]
    * Title: End-to-End Multi-Task Learning with Attention
    * Year: 28 Mar `2018`
    * Authors: Shikun Liu, Edward Johns, Andrew J. Davison
    * Abstract: We propose a novel multi-task learning architecture, which allows learning of task-specific feature-level attention. Our design, the Multi-Task Attention Network (MTAN), consists of a single shared network containing a global feature pool, together with a soft-attention module for each task. These modules allow for learning of task-specific features from the global features, whilst simultaneously allowing for features to be shared across different tasks. The architecture can be trained end-to-end and can be built upon any feed-forward neural network, is simple to implement, and is parameter efficient. We evaluate our approach on a variety of datasets, across both image-to-image predictions and image classification tasks. We show that our architecture is state-of-the-art in multi-task learning compared to existing methods, and is also less sensitive to various weighting schemes in the multi-task loss function. Code is available at this https URL.
* [[Actor-Mimic](https://arxiv.org/abs/1511.06342)]
    [[pdf](https://arxiv.org/pdf/1511.06342.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06342/)]
    * Title: Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning
    * Year: 19 Nov `2015`
    * Authors: Emilio Parisotto, Jimmy Lei Ba, Ruslan Salakhutdinov
    * Abstract: The ability to act in multiple environments and transfer previous knowledge to new situations can be considered a critical aspect of any intelligent agent. Towards this goal, we define a novel method of multitask and transfer learning that enables an autonomous agent to learn how to behave in multiple tasks simultaneously, and then generalize its knowledge to new domains. This method, termed "Actor-Mimic", exploits the use of deep reinforcement learning and model compression techniques to train a single policy network that learns how to act in a set of distinct tasks by using the guidance of several expert teachers. We then show that the representations learnt by the deep policy network are capable of generalizing to new tasks with no prior expert guidance, speeding up learning in novel environments. Although our method can in general be applied to a wide range of problems, we use Atari games as a testing environment to demonstrate these methods.
* [[Policy Distillation](https://arxiv.org/abs/1511.06295)]
    [[pdf](https://arxiv.org/pdf/1511.06295.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1511.06295/)]
    * Title: Policy Distillation
    * Year: 19 Nov `2015`
    * Authors: Andrei A. Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, Raia Hadsell
    * Abstract: Policies for complex visual tasks have been successfully learned with deep reinforcement learning, using an approach called deep Q-networks (DQN), but relatively large (task-specific) networks and extensive training are needed to achieve good performance. In this work, we present a novel method called policy distillation that can be used to extract the policy of a reinforcement learning agent and train a new network that performs at the expert level while being dramatically smaller and more efficient. Furthermore, the same method can be used to consolidate multiple task-specific policies into a single policy. We demonstrate these claims using the Atari domain and show that the multi-task distilled agent outperforms the single-task teachers as well as a jointly-trained DQN agent.
* [[Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)]
    [[pdf](https://arxiv.org/pdf/1810.04650.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1810.04650/)]
    * Title: Multi-Task Learning as Multi-Objective Optimization
    * Year: 10 Oct `2018`
    * Authors: Ozan Sener, Vladlen Koltun
    * Abstract: In multi-task learning, multiple tasks are solved jointly, sharing inductive bias between them. Multi-task learning is inherently a multi-objective problem because different tasks may conflict, necessitating a trade-off. A common compromise is to optimize a proxy objective that minimizes a weighted linear combination of per-task losses. However, this workaround is only valid when the tasks do not compete, which is rarely the case. In this paper, we explicitly cast multi-task learning as multi-objective optimization, with the overall objective of finding a Pareto optimal solution. To this end, we use algorithms developed in the gradient-based multi-objective optimization literature. These algorithms are not directly applicable to large-scale learning problems since they scale poorly with the dimensionality of the gradients and the number of tasks. We therefore propose an upper bound for the multi-objective loss and show that it can be optimized efficiently. We further prove that optimizing this upper bound yields a Pareto optimal solution under realistic assumptions. We apply our method to a variety of multi-task deep learning problems including digit classification, scene understanding (joint semantic segmentation, instance segmentation, and depth estimation), and multi-label classification. Our method produces higher-performing models than recent multi-task learning formulations or per-task training.
