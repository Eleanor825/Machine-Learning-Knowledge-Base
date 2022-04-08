# Learning Rate Schedules

## References

* [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)

## Step Decay

## Cyclical Learning Rate Schedules
    * "Increasing the learning rate might have a short term negative effect and yet achieve a longer term beneficial effect"
    * To gain intuition on why this short-term effect would yield a long-term positive effect, it's important to understand the desirable characteristics of our converged minimum. Ultimately, we'd like our network to learn from the data in a manner which generalizes to unseen data. Further, a network with good generalization properties should be robust in the sense that small changes to the network's parameters don't cause drastic changes to performance. With this in mind, it makes sense that sharp minima lead to poor generalization as small changes to the parameter values may result in a drastically higher loss. By allowing for our learning rate to increase at times, we can "jump out" of sharp minima which would temporarily increase our loss but may ultimately lead to convergence on a more desirable minima.
    * Additionally, increasing the learning rate can also allow for "more rapid traversal of saddle point plateaus.
    * triangular schedule
    * triangular schedule with fixed decay
    * triangular schedule with exponential decay

## Stochastic Gradient Descent with Restarts

## Adaptive Learning Rate
