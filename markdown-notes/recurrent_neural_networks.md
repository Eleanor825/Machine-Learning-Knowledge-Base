## Vanilla RNN

* one single hidden vector $h$.
* computation:
    * $a(t) = W \cdot h(t-1) + U \cdot x(t) + b$.
    * $h(t) = \tanh(a(t))$.
    * $o(t) = V \cdot h(t) + c$.
    * $\hat{y}(t) = \operatorname{softmax}(o(t))$.


## Examples

* one to many: image captioning
* many to one: sentiment classification
* many to many: machine translation


## Parameter Sharing

* Update rule: $h(t) = f(h(t-1),x(t); \theta)$.
    * $h(t-1)$: the previous state at time step $t-1$, memory.
    * $x(t)$: the current input at time step $t$.
    * $\theta$: the parameter, shared across different time steps.
    * same function $f$ used for every time step.


## Training

* back-propagation through time


## Gradient Problems

* common solution: clipping gradient
* Long Short Term Memory (LSTM)
    * 3 gates: $i$, $f$, and $o$.
    * f: forget gate, whether to erase cell
    * i: input gate, whether to write to cell
    * g: gate gate, how much to write to cell
    * o: output gate, how much to reveal cell
    * $c_{t} = f \odot c_{t-1} + i \odot g$.
    * $h_{t} = o \odot \tanh(c_{t})$.
* Gated Recurrent Unit (GRU)
    * 2 gates: $r$ and $Z$.
    * $r_{t} = \sigma(W_{xr}x_{t} + W_{hr}h_{t-1} + b_{r})$.
    * $\tilde{h}(t) = \tanh(W_{xh}x_{t} + W_{hh}(r_{t} \odot h_{t-1}) + b_{h})$.
    * $z_{t} = \sigma(W_{xz}x_{t} + W_{hz}h_{t-1} + b_{z})$.
    * $h_{t} = z_{t} \odot h_{t-1} + (1-z_{t}) \odot \tilde{h}(t)$.
