# [Papers] Explainable AI <!-- omit in toc -->

count=21

## Table of Contents <!-- omit in toc -->

- [Surveys](#surveys)
- [Class Activation Maps](#class-activation-maps)
- [Saliency Map](#saliency-map)
- [Attention Mechanism](#attention-mechanism)
- [From the VoG Paper](#from-the-vog-paper)
- [Others](#others)

----------------------------------------------------------------------------------------------------

## Surveys

* [[Opportunities and Challenges in Explainable Artificial Intelligence (XAI): A Survey](https://arxiv.org/abs/2006.11371)]
    [[pdf](https://arxiv.org/pdf/2006.11371.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2006.11371/)]
    * Title: Opportunities and Challenges in Explainable Artificial Intelligence (XAI): A Survey
    * Year: 16 Jun `2020`
    * Authors: Arun Das, Paul Rad
    * Abstract: Nowadays, deep neural networks are widely used in mission critical systems such as healthcare, self-driving vehicles, and military which have direct impact on human lives. However, the black-box nature of deep neural networks challenges its use in mission critical applications, raising ethical and judicial concerns inducing lack of trust. Explainable Artificial Intelligence (XAI) is a field of Artificial Intelligence (AI) that promotes a set of tools, techniques, and algorithms that can generate high-quality interpretable, intuitive, human-understandable explanations of AI decisions. In addition to providing a holistic view of the current XAI landscape in deep learning, this paper provides mathematical summaries of seminal work. We start by proposing a taxonomy and categorizing the XAI techniques based on their scope of explanations, methodology behind the algorithms, and explanation level or usage which helps build trustworthy, interpretable, and self-explanatory deep learning models. We then describe the main principles used in XAI research and present the historical timeline for landmark studies in XAI from 2007 to 2020. After explaining each category of algorithms and approaches in detail, we then evaluate the explanation maps generated by eight XAI algorithms on image data, discuss the limitations of this approach, and provide potential future directions to improve XAI evaluation.
* [[Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)](https://ieeexplore.ieee.org/document/8466590)]
    * Title: Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)
    * Year: 16 Sep `2018`
    * Authors: Amina Adadi; Mohammed Berrada
    * Abstract: At the dawn of the fourth industrial revolution, we are witnessing a fast and widespread adoption of artificial intelligence (AI) in our daily life, which contributes to accelerating the shift towards a more algorithmic society. However, even with such unprecedented advancements, a key impediment to the use of AI-based systems is that they often lack transparency. Indeed, the black-box nature of these systems allows powerful predictions, but it cannot be directly explained. This issue has triggered a new debate on explainable AI (XAI). A research field holds substantial promise for improving trust and transparency of AI-based systems. It is recognized as the sine qua non for AI to continue making steady progress without disruption. This survey provides an entry point for interested researchers and practitioners to learn key aspects of the young and rapidly growing body of research related to XAI. Through the lens of the literature, we review the existing approaches regarding the topic, discuss trends surrounding its sphere, and present major research trajectories.

## Class Activation Maps

* [[CAM](https://arxiv.org/abs/1512.04150)]
    [[pdf](https://arxiv.org/pdf/1512.04150.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1512.04150/)]
    * Title: Learning Deep Features for Discriminative Localization
    * Year: 14 Dec `2015`
    * Authors: Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba
    * Abstract: In this work, we revisit the global average pooling layer proposed in [13], and shed light on how it explicitly enables the convolutional neural network to have remarkable localization ability despite being trained on image-level labels. While this technique was previously proposed as a means for regularizing training, we find that it actually builds a generic localizable deep representation that can be applied to a variety of tasks. Despite the apparent simplicity of global average pooling, we are able to achieve 37.1% top-5 error for object localization on ILSVRC 2014, which is remarkably close to the 34.2% top-5 error achieved by a fully supervised CNN approach. We demonstrate that our network is able to localize the discriminative image regions on a variety of tasks despite not being trained for them
* [[Grad-CAM](https://arxiv.org/abs/1610.02391)]
    [[pdf](https://arxiv.org/pdf/1610.02391.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1610.02391/)]
    * Title: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    * Year: 07 Oct `2016`
    * Authors: Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra
    * Abstract: We propose a technique for producing "visual explanations" for decisions from a large class of CNN-based models, making them more transparent. Our approach - Gradient-weighted Class Activation Mapping (Grad-CAM), uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept. Grad-CAM is applicable to a wide variety of CNN model-families: (1) CNNs with fully-connected layers, (2) CNNs used for structured outputs, (3) CNNs used in tasks with multimodal inputs or reinforcement learning, without any architectural changes or re-training. We combine Grad-CAM with fine-grained visualizations to create a high-resolution class-discriminative visualization and apply it to off-the-shelf image classification, captioning, and visual question answering (VQA) models, including ResNet-based architectures. In the context of image classification models, our visualizations (a) lend insights into their failure modes, (b) are robust to adversarial images, (c) outperform previous methods on localization, (d) are more faithful to the underlying model and (e) help achieve generalization by identifying dataset bias. For captioning and VQA, we show that even non-attention based models can localize inputs. We devise a way to identify important neurons through Grad-CAM and combine it with neuron names to provide textual explanations for model decisions. Finally, we design and conduct human studies to measure if Grad-CAM helps users establish appropriate trust in predictions from models and show that Grad-CAM helps untrained users successfully discern a 'stronger' nodel from a 'weaker' one even when both make identical predictions. Our code is available at this https URL, along with a demo at this http URL, and a video at this http URL.
* [[Grad-CAM++](https://arxiv.org/abs/1710.11063)]
    [[pdf](https://arxiv.org/pdf/1710.11063.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1710.11063/)]
    * Title: Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
    * Year: 30 Oct `2017`
    * Authors: Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian
    * Abstract: Over the last decade, Convolutional Neural Network (CNN) models have been highly successful in solving complex vision problems. However, these deep models are perceived as "black box" methods considering the lack of understanding of their internal functioning. There has been a significant recent interest in developing explainable deep learning models, and this paper is an effort in this direction. Building on a recently proposed method called Grad-CAM, we propose a generalized method called Grad-CAM++ that can provide better visual explanations of CNN model predictions, in terms of better object localization as well as explaining occurrences of multiple object instances in a single image, when compared to state-of-the-art. We provide a mathematical derivation for the proposed method, which uses a weighted combination of the positive partial derivatives of the last convolutional layer feature maps with respect to a specific class score as weights to generate a visual explanation for the corresponding class label. Our extensive experiments and evaluations, both subjective and objective, on standard datasets showed that Grad-CAM++ provides promising human-interpretable visual explanations for a given CNN architecture across multiple tasks including classification, image caption generation and 3D action recognition; as well as in new settings such as knowledge distillation.
* check this README.md for more techniques in the CAM family https://github.com/jacobgil/pytorch-grad-cam

## Saliency Map

* [[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)]
    [[pdf](https://arxiv.org/pdf/1312.6034.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1312.6034/)]
    * Title: Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
    * Year: 20 Dec `2013`
    * Authors: Karen Simonyan, Andrea Vedaldi, Andrew Zisserman
    * Abstract: This paper addresses the visualisation of image classification models, learnt using deep Convolutional Networks (ConvNets). We consider two visualisation techniques, based on computing the gradient of the class score with respect to the input image. The first one generates an image, which maximises the class score [Erhan et al., 2009], thus visualising the notion of the class, captured by a ConvNet. The second technique computes a class saliency map, specific to a given image and class. We show that such maps can be employed for weakly supervised object segmentation using classification ConvNets. Finally, we establish the connection between the gradient-based ConvNet visualisation methods and deconvolutional networks [Zeiler et al., 2013].
* [[Conditional Affordance Learning for Driving in Urban Environments](https://arxiv.org/abs/1806.06498)]
    [[pdf](https://arxiv.org/pdf/1806.06498.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1806.06498/)]
    * Title: Conditional Affordance Learning for Driving in Urban Environments
    * Year: 18 Jun `2018`
    * Authors: Axel Sauer, Nikolay Savinov, Andreas Geiger
    * Abstract: Most existing approaches to autonomous driving fall into one of two categories: modular pipelines, that build an extensive model of the environment, and imitation learning approaches, that map images directly to control outputs. A recently proposed third paradigm, direct perception, aims to combine the advantages of both by using a neural network to learn appropriate low-dimensional intermediate representations. However, existing direct perception approaches are restricted to simple highway situations, lacking the ability to navigate intersections, stop at traffic lights or respect speed limits. In this work, we propose a direct perception approach which maps video input to intermediate representations suitable for autonomous navigation in complex urban environments given high-level directional inputs. Compared to state-of-the-art reinforcement and conditional imitation learning approaches, we achieve an improvement of up to 68 % in goal-directed navigation on the challenging CARLA simulation benchmark. In addition, our approach is the first to handle traffic lights and speed signs by using image-level labels only, as well as smooth car-following, resulting in a significant reduction of traffic accidents in simulation.
* [[Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/abs/1704.07911)]
    [[pdf](https://arxiv.org/pdf/1704.07911.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1704.07911/)]
    * Title: Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car
    * Year: 25 Apr `2017`
    * Authors: Mariusz Bojarski, Philip Yeres, Anna Choromanska, Krzysztof Choromanski, Bernhard Firner, Lawrence Jackel, Urs Muller
    * Abstract: As part of a complete software stack for autonomous driving, NVIDIA has created a neural-network-based system, known as PilotNet, which outputs steering angles given images of the road ahead. PilotNet is trained using road images paired with the steering angles generated by a human driving a data-collection car. It derives the necessary domain knowledge by observing human drivers. This eliminates the need for human engineers to anticipate what is important in an image and foresee all the necessary rules for safe driving. Road tests demonstrated that PilotNet can successfully perform lane keeping in a wide variety of driving conditions, regardless of whether lane markings are present or not. The goal of the work described here is to explain what PilotNet learns and how it makes its decisions. To this end we developed a method for determining which elements in the road image most influence PilotNet's steering decision. Results show that PilotNet indeed learns to recognize relevant objects on the road. In addition to learning the obvious features such as lane markings, edges of roads, and other cars, PilotNet learns more subtle features that would be hard to anticipate and program by engineers, for example, bushes lining the edge of the road and atypical vehicle classes.
* [[OD-XAI](https://www.mdpi.com/2076-3417/12/11/5310)]
    * Title: OD-XAI: Explainable AI-Based Semantic Object Detection for Autonomous Vehicles
    * Year: 03 Mar `2022`
    * Authors: Harsh Mankodiya, Dhairya Jadav, Rajesh Gupta, Sudeep Tanwar, Wei-Chiang Hong, Ravi Sharma
    * Abstract: In recent years, artificial intelligence (AI) has become one of the most prominent fields in autonomous vehicles (AVs). With the help of AI, the stress levels of drivers have been reduced, as most of the work is executed by the AV itself. With the increasing complexity of models, explainable artificial intelligence (XAI) techniques work as handy tools that allow naive people and developers to understand the intricate workings of deep learning models. These techniques can be paralleled to AI to increase their interpretability. One essential task of AVs is to be able to follow the road. This paper attempts to justify how AVs can detect and segment the road on which they are moving using deep learning (DL) models. We trained and compared three semantic segmentation architectures for the task of pixel-wise road detection. Max IoU scores of 0.9459 and 0.9621 were obtained on the train and test set. Such DL algorithms are called “black box models” as they are hard to interpret due to their highly complex structures. Integrating XAI enables us to interpret and comprehend the predictions of these abstract models. We applied various XAI methods and generated explanations for the proposed segmentation model for road detection in AVs.
* [[Raising context awareness in motion forecasting](https://arxiv.org/abs/2109.08048)]
    [[pdf](https://arxiv.org/pdf/2109.08048.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2109.08048/)]
    * Title: Raising context awareness in motion forecasting
    * Year: 16 Sep `2021`
    * Authors: Hédi Ben-Younes, Éloi Zablocki, Mickaël Chen, Patrick Pérez, Matthieu Cord
    * Abstract: Learning-based trajectory prediction models have encountered great success, with the promise of leveraging contextual information in addition to motion history. Yet, we find that state-of-the-art forecasting methods tend to overly rely on the agent's current dynamics, failing to exploit the semantic contextual cues provided at its input. To alleviate this issue, we introduce CAB, a motion forecasting model equipped with a training procedure designed to promote the use of semantic contextual information. We also introduce two novel metrics - dispersion and convergence-to-range - to measure the temporal consistency of successive forecasts, which we found missing in standard metrics. Our method is evaluated on the widely adopted nuScenes Prediction benchmark as well as on a subset of the most difficult examples from this benchmark. The code is available at this http URL
* [[Explainable AI in Scene Understanding for Autonomous Vehicles in Unstructured Traffic Environments on Indian Roads Using the Inception U-Net Model with Grad-CAM Visualization](https://www.mdpi.com/1424-8220/22/24/9677)]
    * Title: Explainable AI in Scene Understanding for Autonomous Vehicles in Unstructured Traffic Environments on Indian Roads Using the Inception U-Net Model with Grad-CAM Visualization
    * Year: 10 Oct `2022`
    * Authors: Stefano Quer, Ikhlas Abdel-Qader, Suresh Kolekar, Shilpa Gite, Biswajeet Pradhan, Abdullah Alamri
    * Abstract: The intelligent transportation system, especially autonomous vehicles, has seen a lot of interest among researchers owing to the tremendous work in modern artificial intelligence (AI) techniques, especially deep neural learning. As a result of increased road accidents over the last few decades, significant industries are moving to design and develop autonomous vehicles. Understanding the surrounding environment is essential for understanding the behavior of nearby vehicles to enable the safe navigation of autonomous vehicles in crowded traffic environments. Several datasets are available for autonomous vehicles focusing only on structured driving environments. To develop an intelligent vehicle that drives in real-world traffic environments, which are unstructured by nature, there should be an availability of a dataset for an autonomous vehicle that focuses on unstructured traffic environments. Indian Driving Lite dataset (IDD-Lite), focused on an unstructured driving environment, was released as an online competition in NCPPRIPG 2019. This study proposed an explainable inception-based U-Net model with Grad-CAM visualization for semantic segmentation that combines an inception-based module as an encoder for automatic extraction of features and passes to a decoder for the reconstruction of the segmentation feature map. The black-box nature of deep neural networks failed to build trust within consumers. Grad-CAM is used to interpret the deep-learning-based inception U-Net model to increase consumer trust. The proposed inception U-net with Grad-CAM model achieves 0.622 intersection over union (IoU) on the Indian Driving Dataset (IDD-Lite), outperforming the state-of-the-art (SOTA) deep neural-network-based segmentation models.

## Attention Mechanism

* [[Show, Attend and Tell](https://arxiv.org/abs/1502.03044)]
    [[pdf](https://arxiv.org/pdf/1502.03044.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1502.03044/)]
    * Title: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
    * Year: 10 Feb `2015`
    * Authors: Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio
    * Abstract: Inspired by recent work in machine translation and object detection, we introduce an attention based model that automatically learns to describe the content of images. We describe how we can train this model in a deterministic manner using standard backpropagation techniques and stochastically by maximizing a variational lower bound. We also show through visualization how the model is able to automatically learn to fix its gaze on salient objects while generating the corresponding words in the output sequence. We validate the use of attention with state-of-the-art performance on three benchmark datasets: Flickr8k, Flickr30k and MS COCO.
* [[Interpretable Learning for Self-Driving Cars by Visualizing Causal Attention](https://arxiv.org/abs/1703.10631)]
    [[pdf](https://arxiv.org/pdf/1703.10631.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.10631/)]
    * Title: Interpretable Learning for Self-Driving Cars by Visualizing Causal Attention
    * Year: 30 Mar `2017`
    * Authors: Jinkyu Kim, John Canny
    * Abstract: Deep neural perception and control networks are likely to be a key component of self-driving vehicles. These models need to be explainable - they should provide easy-to-interpret rationales for their behavior - so that passengers, insurance companies, law enforcement, developers etc., can understand what triggered a particular behavior. Here we explore the use of visual explanations. These explanations take the form of real-time highlighted regions of an image that causally influence the network's output (steering control). Our approach is two-stage. In the first stage, we use a visual attention model to train a convolution network end-to-end from images to steering angle. The attention model highlights image regions that potentially influence the network's output. Some of these are true influences, but some are spurious. We then apply a causal filtering step to determine which input regions actually influence the output. This produces more succinct visual explanations and more accurately exposes the network's behavior. We demonstrate the effectiveness of our model on three datasets totaling 16 hours of driving. We first show that training with attention does not degrade the performance of the end-to-end network. Then we show that the network causally cues on a variety of features that are used by humans while driving.
* [[Attentional Bottleneck](https://arxiv.org/abs/2005.04298)]
    [[pdf](https://arxiv.org/pdf/2005.04298.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2005.04298/)]
    * Title: Attentional Bottleneck: Towards an Interpretable Deep Driving Network
    * Year: 08 May `2020`
    * Authors: Jinkyu Kim, Mayank Bansal
    * Abstract: Deep neural networks are a key component of behavior prediction and motion generation for self-driving cars. One of their main drawbacks is a lack of transparency: they should provide easy to interpret rationales for what triggers certain behaviors. We propose an architecture called Attentional Bottleneck with the goal of improving transparency. Our key idea is to combine visual attention, which identifies what aspects of the input the model is using, with an information bottleneck that enables the model to only use aspects of the input which are important. This not only provides sparse and interpretable attention maps (e.g. focusing only on specific vehicles in the scene), but it adds this transparency at no cost to model accuracy. In fact, we find slight improvements in accuracy when applying Attentional Bottleneck to the ChauffeurNet model, whereas we find that the accuracy deteriorates with a traditional visual attention model.

## From the VoG Paper

* [[A Benchmark for Interpretability Methods in Deep Neural Networks](https://arxiv.org/abs/1806.10758)]
    [[pdf](https://arxiv.org/pdf/1806.10758.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1806.10758/)]
    * Title: A Benchmark for Interpretability Methods in Deep Neural Networks
    * Year: 28 Jun `2018`
    * Authors: Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, Been Kim
    * Abstract: We propose an empirical measure of the approximate accuracy of feature importance estimates in deep neural networks. Our results across several large-scale image classification datasets show that many popular interpretability methods produce estimates of feature importance that are not better than a random designation of feature importance. Only certain ensemble based approaches---VarGrad and SmoothGrad-Squared---outperform such a random assignment of importance. The manner of ensembling remains critical, we show that some approaches do no better then the underlying method but carry a far higher computational burden.
* [[Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)]
    [[pdf](https://arxiv.org/pdf/1704.02685.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1704.02685/)]
    * Title: Learning Important Features Through Propagating Activation Differences
    * Year: 10 Apr `2017`
    * Authors: Avanti Shrikumar, Peyton Greenside, Anshul Kundaje
    * Abstract: The purported "black box" nature of neural networks is a barrier to adoption in applications where interpretability is essential. Here we present DeepLIFT (Deep Learning Important FeaTures), a method for decomposing the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons in the network to every feature of the input. DeepLIFT compares the activation of each neuron to its 'reference activation' and assigns contribution scores according to the difference. By optionally giving separate consideration to positive and negative contributions, DeepLIFT can also reveal dependencies which are missed by other approaches. Scores can be computed efficiently in a single backward pass. We apply DeepLIFT to models trained on MNIST and simulated genomic data, and show significant advantages over gradient-based methods. Video tutorial: this http URL, ICML slides: this http URL, ICML talk: this https URL, code: this http URL.
* [[SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)]
    [[pdf](https://arxiv.org/pdf/1706.03825.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1706.03825/)]
    * Title: SmoothGrad: removing noise by adding noise
    * Year: 12 Jun `2017`
    * Authors: Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg
    * Abstract: Explaining the output of a deep network remains a challenge. In the case of an image classifier, one type of explanation is to identify pixels that strongly influence the final decision. A starting point for this strategy is the gradient of the class score function with respect to the input image. This gradient can be interpreted as a sensitivity map, and there are several techniques that elaborate on this basic idea. This paper makes two contributions: it introduces SmoothGrad, a simple method that can help visually sharpen gradient-based sensitivity maps, and it discusses lessons in the visualization of these maps. We publish the code for our experiments and a website with our results.
* [[Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)]
    [[pdf](https://arxiv.org/pdf/1703.01365.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1703.01365/)]
    * Title: Axiomatic Attribution for Deep Networks
    * Year: 04 Mar `2017`
    * Authors: Mukund Sundararajan, Ankur Taly, Qiqi Yan
    * Abstract: We study the problem of attributing the prediction of a deep network to its input features, a problem previously studied by several other works. We identify two fundamental axioms---Sensitivity and Implementation Invariance that attribution methods ought to satisfy. We show that they are not satisfied by most known attribution methods, which we consider to be a fundamental weakness of those methods. We use the axioms to guide the design of a new attribution method called Integrated Gradients. Our method requires no modification to the original network and is extremely simple to implement; it just needs a few calls to the standard gradient operator. We apply this method to a couple of image models, a couple of text models and a chemistry model, demonstrating its ability to debug networks, to extract rules from a network, and to enable users to engage with models better.

## Others

* [[Towards Explainable Motion Prediction using Heterogeneous Graph Representations](https://arxiv.org/abs/2212.03806)]
    [[pdf](https://arxiv.org/pdf/2212.03806.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2212.03806/)]
    * Title: Towards Explainable Motion Prediction using Heterogeneous Graph Representations
    * Year: 07 Dec `2022`
    * Authors: Sandra Carrasco Limeros, Sylwia Majchrowska, Joakim Johnander, Christoffer Petersson, David Fernández Llorca
    * Abstract: Motion prediction systems aim to capture the future behavior of traffic scenarios enabling autonomous vehicles to perform safe and efficient planning. The evolution of these scenarios is highly uncertain and depends on the interactions of agents with static and dynamic objects in the scene. GNN-based approaches have recently gained attention as they are well suited to naturally model these interactions. However, one of the main challenges that remains unexplored is how to address the complexity and opacity of these models in order to deal with the transparency requirements for autonomous driving systems, which includes aspects such as interpretability and explainability. In this work, we aim to improve the explainability of motion prediction systems by using different approaches. First, we propose a new Explainable Heterogeneous Graph-based Policy (XHGP) model based on an heterograph representation of the traffic scene and lane-graph traversals, which learns interaction behaviors using object-level and type-level attention. This learned attention provides information about the most important agents and interactions in the scene. Second, we explore this same idea with the explanations provided by GNNExplainer. Third, we apply counterfactual reasoning to provide explanations of selected individual scenarios by exploring the sensitivity of the trained model to changes made to the input data, i.e., masking some elements of the scene, modifying trajectories, and adding or removing dynamic agents. The explainability analysis provided in this paper is a first step towards more transparent and reliable motion prediction systems, important from the perspective of the user, developers and regulatory agencies. The code to reproduce this work is publicly available at this https URL.
* [[Explaining Autonomous Driving by Learning End-to-End Visual Attention](https://arxiv.org/abs/2006.03347)]
    [[pdf](https://arxiv.org/pdf/2006.03347.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/2006.03347/)]
    * Title: Explaining Autonomous Driving by Learning End-to-End Visual Attention
    * Year: 05 Jun `2020`
    * Authors: Luca Cultrera, Lorenzo Seidenari, Federico Becattini, Pietro Pala, Alberto Del Bimbo
    * Abstract: Current deep learning based autonomous driving approaches yield impressive results also leading to in-production deployment in certain controlled scenarios. One of the most popular and fascinating approaches relies on learning vehicle controls directly from data perceived by sensors. This end-to-end learning paradigm can be applied both in classical supervised settings and using reinforcement learning. Nonetheless the main drawback of this approach as also in other learning problems is the lack of explainability. Indeed, a deep network will act as a black-box outputting predictions depending on previously seen driving patterns without giving any feedback on why such decisions were taken. While to obtain optimal performance it is not critical to obtain explainable outputs from a learned agent, especially in such a safety critical field, it is of paramount importance to understand how the network behaves. This is particularly relevant to interpret failures of such systems. In this work we propose to train an imitation learning based agent equipped with an attention model. The attention model allows us to understand what part of the image has been deemed most important. Interestingly, the use of attention also leads to superior performance in a standard benchmark using the CARLA driving simulator.
* [[Deep Object-Centric Policies for Autonomous Driving](https://arxiv.org/abs/1811.05432)]
    [[pdf](https://arxiv.org/pdf/1811.05432.pdf)]
    [[vanity](https://www.arxiv-vanity.com/papers/1811.05432/)]
    * Title: Deep Object-Centric Policies for Autonomous Driving
    * Year: 13 Nov `2018`
    * Authors: Dequan Wang, Coline Devin, Qi-Zhi Cai, Fisher Yu, Trevor Darrell
    * Abstract: While learning visuomotor skills in an end-to-end manner is appealing, deep neural networks are often uninterpretable and fail in surprising ways. For robotics tasks, such as autonomous driving, models that explicitly represent objects may be more robust to new scenes and provide intuitive visualizations. We describe a taxonomy of "object-centric" models which leverage both object instances and end-to-end learning. In the Grand Theft Auto V simulator, we show that object-centric models outperform object-agnostic methods in scenes with other vehicles and pedestrians, even with an imperfect detector. We also demonstrate that our architectures perform well on real-world environments by evaluating on the Berkeley DeepDrive Video dataset, where an object-centric model outperforms object-agnostic models in the low-data regimes.
