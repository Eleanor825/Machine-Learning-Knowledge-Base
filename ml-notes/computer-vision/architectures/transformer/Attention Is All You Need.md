# Attention Is All You Need

* Year: 12 Jun `2017`
* Author: Ashish Vaswani
* Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

----------------------------------------------------------------------------------------------------

## 1 Introduction

## 2 Background

## 3 Model Architecture

> Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbol representations $\textbf{x}$ to a sequence of continuous representations $\textbf{z}$. Given $\textbf{z}$, the decoder then generates an output sequence $\textbf{y}$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

> The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

### 3.1 Encoder and Decoder Stacks

> **Encoder:** The encoder is composed of a stack of $N = 6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is $\text{LayerNorm}(x + \text{Sublayer}(x))$, where $\text{Sublayer}(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}} = 512$.

> **Decoder:** The decoder is also composed ofa stack of $N = 6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sublayers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

### 3.2 Attention

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

#### 3.2.1 Scaled Dot-Product Attention

> The input consists of queries and keys of dimension $d_{k}$, and values of dimension $d_{v}$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_{k}}$, and apply a softmax function to obtain the weights on the values.

Notations:
* Let $d_{k}$ denote the dimension of the queries and keys.
* Let $d_{v}$ denote the dimension of the values.
* Let $m$ denote the number of queries.
* Let $n$ denote the number of key-value pairs.
* Let $Q \in \mathbb{R}^{m \times d_{k}}$ denote the matrix of queries.
* Let $K \in \mathbb{R}^{n \times d_{k}}$ denote the matrix of keys.
* Let $V \in \mathbb{R}^{n \times d_{v}}$ denote the matrix of values.

Then the Scaled Dot-Product Attention layer $\text{Attention}: \mathbb{R}^{m \times d_{k}} \oplus \mathbb{R}^{n \times d_{k}} \oplus \mathbb{R}^{n \times d_{v}} \to \mathbb{R}^{m \times d_{v}}$ is given by

$$\text{Attention}(Q, K, V) = \text{softmax}\bigg(\frac{QK^{\top}}{\sqrt{d_{k}}}\bigg)V.$$

> Two most commonly used attention functions are `additive` attention, and dot-product (`multiplicative`) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_{k}}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

> While for small values of $d_{k}$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_{k}$. We suspect that for large values of $d{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{d_{k}}$.

#### 3.2.2 Multi-Head Attention

> Instead of performing a single attention function with $d_{\text{model}}$-dimensional keys, values, and queries, we found it beneficial to linearly project the queries, keys, and values $h$ times with different, learned linear projections to $d_{k}$, $d_{k}$, and $d_{v}$ dimensions, respectively. On each of these projected versions of queries, keys, and values we then perform the attention function in parallel, yielding $d_{v}$-dimensional output values. These are concatenated and once again projected, resulting in the final values.

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

Notations:
* Let $m$ denote the number of queries.
* Let $n$ denote the number of key-value pairs.
* Let $Q \in \mathbb{R}^{m \times d_{\text{model}}}$ denote the matrix of queries.
* Let $K \in \mathbb{R}^{n \times d_{\text{model}}}$ denote the matrix of keys.
* Let $V \in \mathbb{R}^{n \times d_{\text{model}}}$ denote the matrix of values.
* Let $W_{i}^{Q} \in \mathbb{R}^{d_{\text{model}} \times d_{k}}$ denote the projection operation for $Q$, $\forall i \in \{1, ..., h\}$.
* Let $W_{i}^{K} \in \mathbb{R}^{d_{\text{model}} \times d_{k}}$ denote the projection operation for $K$, $\forall i \in \{1, ..., h\}$.
* Let $W_{i}^{V} \in \mathbb{R}^{d_{\text{model}} \times d_{v}}$ denote the projection operation for $V$, $\forall i \in \{1, ..., h\}$.
* Let $W^{O} \in \mathbb{R}^{hd_{v} \times d_{\text{model}}}$ denote the projection operation on the heads.

Then the Multi-Head Attention layer $\text{MultiHead}: \mathbb{R}^{m \times d_{\text{model}}} \oplus \mathbb{R}^{n \times d_{\text{model}}} \oplus \mathbb{R}^{n \times d_{\text{model}}} \to \mathbb{R}^{m \times d_{\text{model}}}$ is given by

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_{1}, ..., \text{head}_{h})W^{O}$$
where $\forall i \in \{1, ..., h\}$,
$$\text{head}_{i} := \text{Attention}(QW_{i}^{Q}, KW{i}^{K}, VW{i}^{V}).$$

> In this work we employ $h = 8$ parallel attention layers, or heads. For each  of these we use $d_{k} = d_{v} = d_{\text{model}} / h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single -head attention with full dimensionality.

#### 3.2.3 Applications of Attention in our Model

> The Transformer uses multi-head attention in three different ways:
> * In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models.
> * The encoder contains self-attention layers. In a self-attention layer all of the keys, values, and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the  previous layer of the encoder.
> * Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot--product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections.

### 3.3 Position-wise Feed-Forward Networks

> In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
$$\text{FFN}(x) = \max(0, xW_{1} + b_{1})W_{2} + b_{2}.$$

### 3.4 Embeddings and Softmax

### 3.5 Positional Encoding

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms oft he encoder and decoder stacks. The positional encodings have the same dimension $d_{\text{model}}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed.

> In this work, we use sine and cosine functions of different frequencies:
> $$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d_{\text{model}}})$$
> $$\text{PE}(pos,2i+1) = \cos(pos / 10000^{2i/d_{\text{model}}})$$
> where $pos$ is the position and $i$ is the dimension. That iss each dimension of the positional encoding corresponds to a sinusoid. Thee wavelengths form a geometricc progression from $2\pi$ to $10000 \cdot 2\pi$. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $\text{PE}(pos + k)$ can be represented as a linear function of $\text{PE}(pos)$.

> We also experimented with using learned positional embeddings instead, and found that the two versions produced nearly identical results. We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

## Why Self-Attention

