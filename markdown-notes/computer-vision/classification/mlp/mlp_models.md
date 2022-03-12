# MLP Models

## MLP-Mixer

* Modern deep vision architectures consist of layers that mix features
    * at a given spatial location
    * between different spatial locations
    * both
* The idea behind the Mixer architecture is to clearly separate the per-location (channel-mixing) operations and cross-location (token-mixing) operations.
  Both operations are implemented with MLPs.
* A MLP-Mixer model contains multiple mixer blocks and a prediction head.
* Each `MixerBlock` contains:
    1. Token-mixing layer
        * Allow communication between different spatial locations (tokens).
        * Operate on each channel independently.
        * Take individual columns of the table as inputs.
        * $\mathbb{R}^{S} \to \mathbb{R}^{S}$.
    $$\mathbf{U}_{*,i} = \underbrace{\;\mathbf{X}_{*,i}\;}_{\text{skip connection}} + \mathbf{W}_{2}\sigma(\mathbf{W}_{1}\text{LayerNorm}(\mathbf{X})_{*,i}), \quad \text{ for } i = 1...C$$
    2. Channel-mixing layer
        * Allow communication between different channels.
        * Operate on each token independently.
        * Take individual rows of the table as inputs.
        * $\mathbb{R}^{C} \to \mathbb{R}^{C}$.
    $$\mathbf{Y}_{j,*} = \underbrace{\;\mathbf{U}_{j,*}\;}_{\text{skip connection}} + \mathbf{W}_{4}\sigma(\mathbf{W}_{3}\text{LayerNorm}(\mathbf{U})_{j,*}), \quad \text{ for } j = 1...S$$
* Each mixing layer contains:
    1. `LayerNorm`
    2. `MLPBlock`  
        (1) Fully connected layer  
        (2) Non-linearity (GELU)  
        (3) Fully connected layer  
