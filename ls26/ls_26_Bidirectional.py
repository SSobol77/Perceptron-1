import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, GRU, Input, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Generate synthetic data (sine wave with noise)
N = 10000
data = np.array([np.sin(x/20) for x in range(N)]) + 0.1*np.random.randn(N)
plt.plot(data[:100])  # Plot the first 100 data points

off = 3
length = off*2+1
# Create input data X and target data Y
X = np.array([np.diag(np.hstack((data[i:i+off], data[i+off+1:i+length]))) for i in range(N-length)])
Y = data[off:N-off-1]
print(X.shape, Y.shape, sep='\n')  # Print the shapes of X and Y

# Build the Bidirectional GRU model
model = Sequential()
model.add(Input((length-1, length-1)))
model.add(Bidirectional(GRU(2)))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(0.01))

# Train the model using the data X and target Y
history = model.fit(X, Y, batch_size=32, epochs=10)

M = 200
XX = np.zeros(M)
XX[:off] = data[:off]
# Generate new data points using the trained model
for i in range(M-off-1):
    x = np.diag(np.hstack((XX[i:i+off], data[i+off+1:i+length])))
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    XX[i+off] = y

# Plot the generated data and original data
plt.plot(XX[:M], label='Generated Data')
plt.plot(data[:M], label='Original Data')
plt.legend()
plt.show()
