# Perceptron-1 
## neural network
Creating a basic artificial neural network model.
Algorithm of Back Propagation Through Time (BPTT)

Let's unfold the recurrent neural network over three time steps. At each moment in time, a vector x is input to it, and the output value y is observed at the third step. To train such a network, the back propagation algorithm can be used, taking into account the temporal nature of the network's behavior. In our case, the recurrent network is built according to the "many to one" principle, where there are multiple input signals and one output. Since we are dealing with a classification task, we will choose the softmax activation function for the output neurons, and losses will be computed using cross-entropy.
