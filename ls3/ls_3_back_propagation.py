import numpy as np

# Activation function (Hyperbolic Tangent)
def f(x):
    return 2/(1 + np.exp(-x)) - 1

# Derivative of the activation function
def df(x):
    return 0.5*(1 + x)*(1 - x)

# Define the weights for the first and second layers
W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
W2 = np.array([0.2, 0.3])

# Forward propagation through the neural network
def go_forward(inp):
    sum = np.dot(W1, inp)
    out = np.array([f(x) for x in sum])

    sum = np.dot(W2, out)
    y = f(sum)
    return (y, out)

# Training the neural network using backpropagation
def train(epoch):
    global W2, W1
    lmd = 0.01          # learning rate
    N = 10000           # number of iterations for training
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]  # randomly choose an input signal from the training set
        y, out = go_forward(x[0:3])             # forward pass through the network and calculate neuron outputs
        e = y - x[-1]                           # error
        delta = e * df(y)                       # local gradient
        W2[0] = W2[0] - lmd * delta * out[0]    # weight correction for the first connection
        W2[1] = W2[1] - lmd * delta * out[1]    # weight correction for the second connection

        delta2 = W2 * delta * df(out)           # vector of two local gradient values

        # weight corrections for the first layer connections
        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd

# Training dataset (also the complete dataset)
epoch = [(-1, -1, -1, -1),
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, -1, 1, 1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]

train(epoch)        # initiate the network training

# Checking the obtained results
for x in epoch:
    y, out = go_forward(x[0:3])
    print(f"Network Output: {y} => {x[-1]}")
