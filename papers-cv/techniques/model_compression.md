<span style="font-family:monospace">

# Papers in Model Compression

count: 2

## Knowledge Distillation

* [Model Compression](https://dl.acm.org/doi/10.1145/1150402.1150464)
    * Title: Model compression
    * Year: 20 Aug `2006`
    * Author: Cristian Bucilua
    * Abstract: Often the best performing supervised learning models are ensembles of hundreds or thousands of base-level classifiers. Unfortunately, the space required to store this many classifiers, and the time required to execute them at run-time, prohibits their use in applications where test sets are large (e.g. Google), where storage space is at a premium (e.g. PDAs), and where computational power is limited (e.g. hea-ring aids). We present a method for "compressing" large, complex ensembles into smaller, faster models, usually without significant loss in performance.
* [Distillation](https://arxiv.org/abs/1503.02531)
    * Title: Distilling the Knowledge in a Neural Network
    * Year: 09 Mar `2015`
    * Author: Geoffrey Hinton
    * Abstract: A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.
