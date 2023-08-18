# This code is an implementation of a simple artificial neural network with two layers. 


import numpy as np        # Importing NumPy library, which is commonly used for numerical operations in Python.

# Activation Function:
def act(x):               # The act() function is a simple activation function that returns 0 if the input x is less than 0.5, and 1 otherwise.
    return 0 if x < 0.5 else 1

# Main Function go:
def go(house, rock, attr):      # This is the main function that takes three input parameters: house, rock, and attr.
    
    # Input and Weights Initialization:
    x = np.array([house, rock, attr])     # 'x' is an array containing the input values.
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])  # weight1 is a 2x3 matrix representing the weights connecting the input layer to the hidden layer. Each row corresponds to a neuron in the hidden layer, and each column corresponds to an input feature.
    weight2 = np.array([-1, 1])     # weight2 is a 1x2 vector representing the weights connecting the hidden layer to the output.

    # Hidden Layer Calculation:  
    sum_hidden = np.dot(weight1, x)       # This computes the weighted sum of inputs at the neurons of the hidden layer.

    print("The values of the sums on the neurons of the hidden layer: "+str(sum_hidden))

    # Activation of Hidden Layer Neurons:
    out_hidden = np.array([act(x) for x in sum_hidden])    # This applies the activation function to the computed sums for the hidden layer neurons.
    print("Values at the outputs of the hidden layer neurons: "+str(out_hidden))

    # Output Layer Calculation and Activation:  
    sum_end = np.dot(weight2, out_hidden)       # 'sum_end' computes the weighted sum of outputs from the hidden layer.
    y = act(sum_end)                            # 'y' is the final output of the neural network after applying the activation function.
    print("Output value of HL: "+str(y))

    # Returning Output: The function returns the final output y.
    return y
  
#Input Values and Function Call. The variables house, rock, and attr are set as input values, and the go function is called with these values:
house = 1
rock = 0
attr = 1

# Output Interpretation: Depending on the output value (res), the code prints either "I like you" or "Let's call each other". 
# This seems to be a playful example of using a simple neural network to simulate a decision based on the provided inputs:
res = go(house, rock, attr)
if res == 1:
    print("I like you")
else:
    print("Let's call you")
