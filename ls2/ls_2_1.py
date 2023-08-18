# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the number of data points
N = 5

# Generate random data points for class C1
x1 = np.random.random(N)
x2 = x1 + [np.random.randint(10)/10 for i in range(N)]
C1 = [x1, x2]

# Generate random data points for class C2
x1 = np.random.random(N)
x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1
C2 = [x1, x2]

# Define the classification boundary
f = [0, 1]

# Define the weight vector
w = np.array([-0.3, 0.3])

# Iterate through each data point in C2
for i in range(N):
    x = np.array([C2[0][i], C2[1][i]])
    y = np.dot(w, x)
    if y >= 0:
        print("Класс C1")  # Print class label C1
    else:
        print("Класс C2")  # Print class label C2

# Create a scatter plot for class C1 (red) and class C2 (blue)
plt.scatter(C1[0][:], C1[1][:], s=10, c='red')
plt.scatter(C2[0][:], C2[1][:], s=10, c='blue')

# Plot the classification boundary
plt.plot(f)

# Add grid lines
plt.grid(True)

# Display the plot
plt.show()
