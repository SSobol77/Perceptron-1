# Perceptron-1 neural network
##  Algorithm of Back Propagation Through Time (BPTT)

Creating a basic artificial neural network model.

The recurrent neural network over three time steps. At each moment in time, a vector x is input to it, and the output value y is observed at the third step. To train such a network, the back propagation algorithm can be used, taking into account the temporal nature of the network's behavior. 
|                                             |                                                  |
|:-------------------------------------------:|:------------------------------------------------:|
| ![img](https://github.com/SSobol77/Perceptron-1/blob/main/cats400.jpg) |![img](https://github.com/SSobol77/Perceptron-1/blob/main/cats400_2.jpg) |
| ![img](https://github.com/SSobol77/Perceptron-1/blob/main/img.jpg) | ![img](https://github.com/SSobol77/Perceptron-1/blob/main/img_style.jpg) |

 &nbsp;
 
In our case, the recurrent network is built according to the "many to one" principle, where there are multiple input signals and one output. Since we are dealing with a classification task, we will choose the softmax activation function for the output neurons, and losses will be computed using cross-entropy.
 
 &nbsp;
 
> Backpropagation Through Time (BPTT) is a training algorithm used in recurrent neural networks (RNNs) and their variants like Long Short-Term Memory (LSTM) networks. 
> It's used to update the network's weights by computing gradients with respect to the loss function over a sequence of time steps. 

 &nbsp;
 
The algorithm unfolds the recurrent network through time, turning it into a feedforward network, and then applies the standard backpropagation algorithm to compute gradients. Here's the basic algorithm for BPTT:

*1. Initialization:*

 * Initialize the network weights and biases randomly or using a specific initialization scheme.
  
 * Set the learning rate and other hyperparameters.
   
   &nbsp;

*2. Input Sequences:*

 * Prepare your input data as a sequence of time steps. Each time step has an input vector.

   &nbsp;

*3. Forward Pass:*

  * For each time step ``t`` in the sequence:
    
    * Compute the hidden state ``h_t`` using the current input x_t and the previous hidden state ``h_{t-1}``.
      
    * Calculate the output of the network ``y_t`` using the current hidden state ``h_t``.
   
      &nbsp;

*4. Loss Computation:*

   * Calculate the loss at each time step using the predicted output ``y_t`` and the corresponding `target or ground truth ``target_t``.

  &nbsp;

*5. Backward Pass Through Time:*

 * Initialize the gradient of the loss with respect to the output layer ``dL/dy_t`` for the last time step.
  
 * For each time step t in reverse order:
   
    * Compute the gradient of the loss with respect to the hidden state ``dL/dh_t = dL/dh_t + dL/dy_t * dy_t/dh_t``.
      
    * Update the gradients of the weights and biases using the gradient of the loss with respect to the hidden state and the inputs.

 &nbsp;
 
*6. Gradient Descent Update:*

 * Use the computed gradients to update the network's weights and biases. This can be done using various optimization algorithms like stochastic gradient descent (SGD), Adam, RMSProp, etc.

 &nbsp;
 
*7. Repeat:*

 * Iterate over the dataset multiple times (epochs), updating the weights after each pass.

&nbsp;

It's important to note that BPTT can suffer from the vanishing gradient problem when training deep recurrent networks over long sequences. This can make it challenging for the network to effectively learn dependencies that span a large number of time steps. 

Using techniques like gradient clipping, using gating mechanisms like LSTMs or Gated Recurrent Units (GRUs), and applying more advanced optimization algorithms can help address some of these issues.

