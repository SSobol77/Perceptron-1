# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the activation function
def act(x):
    return 0 if x <= 0 else 1

# Define the main function for the neural network
def go(C):
    # Prepare the input vector with bias term
    x = np.array([C[0], C[1], 1])
    
    # Define the weights for the hidden layer neurons
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    
    # Define the weights for the output layer neuron
    w_out = np.array([-1, 1, -0.5])

    # Calculate the weighted sum for the hidden layer
    sum_hidden = np.dot(w_hidden, x)
    
    # Apply activation function to hidden layer outputs
    out = [act(x) for x in sum_hidden]
    
    # Append bias term to the hidden layer outputs
    out.append(1)
    out = np.array(out)
    
    # Calculate the weighted sum for the output layer
    sum_output = np.dot(w_out, out)
    
    # Apply activation function to the output
    y = act(sum_output)
    
    return y

# Define data points for classes C1 and C2
C1 = [(1,0), (0,1)]
C2 = [(0,0), (1,1)]

# Print the results of the neural network for the provided data points
print(go(C1[0]), go(C1[1]))
print(go(C2[0]), go(C2[1]))
