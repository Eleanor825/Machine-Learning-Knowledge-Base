# Activation Functions

## Rectified Linear Unit and its Variants
* Rectified Linear Unit (ReLU)
$$ f(x) = \max\{0, x\}.$$
* Leaky Rectified Linear Unit (Leaky ReLU)
$$ f(x) = \max\{0.1x, x\}.$$
* Exponential Linear Unit (ELU)
$$ f(x) = \begin{cases}
    x,               & \text{ if } x \geq 0 \\
    \alpha(e^{x}-1), & \text{ if } x \leq 0.
\end{cases}$$
