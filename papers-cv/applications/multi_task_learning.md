# [Papers] Multi-Task Learning <!-- omit in toc -->

count=1

## Table of Contents <!-- omit in toc -->

- [1. Unclassified](#1-unclassified)

----------------------------------------------------------------------------------------------------

## 1. Unclassified

* [[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)]
    [[pdf](https://arxiv.org/pdf/2001.06782.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2001.06782/)]
    * Title: Gradient Surgery for Multi-Task Learning
    * Year: 19 Jan `2020`
    * Authors: Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn
    * Abstract: While deep learning and deep reinforcement learning (RL) systems have demonstrated impressive results in domains such as image classification, game playing, and robotic control, data efficiency remains a major challenge. Multi-task learning has emerged as a promising approach for sharing structure across multiple tasks to enable more efficient learning. However, the multi-task setting presents a number of optimization challenges, making it difficult to realize large efficiency gains compared to learning tasks independently. The reasons why multi-task learning is so challenging compared to single-task learning are not fully understood. In this work, we identify a set of three conditions of the multi-task optimization landscape that cause detrimental gradient interference, and develop a simple yet general approach for avoiding such interference between task gradients. We propose a form of gradient surgery that projects a task's gradient onto the normal plane of the gradient of any other task that has a conflicting gradient. On a series of challenging multi-task supervised and multi-task RL problems, this approach leads to substantial gains in efficiency and performance. Further, it is model-agnostic and can be combined with previously-proposed multi-task architectures for enhanced performance.
