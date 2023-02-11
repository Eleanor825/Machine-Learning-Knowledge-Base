* Training Very Deep Networks [13][14][15]

* [[Momentum](https://proceedings.mlr.press/v28/sutskever13.html)]
    [[pdf](http://proceedings.mlr.press/v28/sutskever13.pdf)]
    * Title: On the importance of initialization and momentum in deep learning
    * Year: `2013`
    * Authors: Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton
    * Institutions: 
    * Abstract: Deep and recurrent neural networks (DNNs and RNNs respectively) are powerful models that were considered to be almost impossible to train using stochastic gradient descent with momentum. In this paper, we show that when stochastic gradient descent with momentum uses a well-designed random initialization and a particular type of slowly increasing schedule for the momentum parameter, it can train both DNNs and RNNs (on datasets with long-term dependencies) to levels of performance that were previously achievable only with Hessian-Free optimization. We find that both the initialization and the momentum are crucial since poorly initialized networks cannot be trained with momentum and well-initialized networks perform markedly worse when the momentum is absent or poorly tuned. Our success training these models suggests that previous attempts to train deep and recurrent neural networks from random initializations have likely failed due to poor initialization schemes. Furthermore, carefully tuned momentum methods suffice for dealing with the curvature issues in deep and recurrent network training objectives without the need for sophisticated second-order methods.
* [[AdaGrad]()]
    [[pdf]()]
    * Title: Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
    * Year: `2010`
    * Authors: John Duchi , Elad Hazan , Yoram Singer
    * Institions: 
    * Abstract: Stochastic subgradient methods are widely used, well analyzed, and constitute effective tools for optimization and online learning. Stochastic gradient methods ’ popularity and appeal are largely due to their simplicity, as they largely follow predetermined procedural schemes. However, most common subgradient approaches are oblivious to the characteristics of the data being observed. We present a new family of subgradient methods that dynamically incorporate knowledge of the geometry of the data observed in earlier iterations to perform more informative gradient-based learning. The adaptation, in essence, allows us to find needles in haystacks in the form of very predictive but rarely seenfeatures. Ourparadigmstemsfromrecentadvancesinstochasticoptimizationandonlinelearning which employ proximal functions to control the gradient steps of the algorithm. We describe and analyze an apparatus for adaptively modifying the proximal function, which significantly simplifies setting a learning rate and results in regret guarantees that are provably as good as the best proximal function that can be chosen in hindsight. In a companion paper, we validate experimentally our theoretical analysis and show that the adaptive subgradient approach outperforms state-of-the-art, but non-adaptive, subgradient algorithms.
