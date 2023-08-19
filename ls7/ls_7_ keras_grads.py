import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Input data
c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

# Create a sequential model
model = keras.Sequential()

# Add a linear activation dense layer to the model
model.add(Dense(units=1, input_shape=(1,), activation='linear'))

# Compile the model with mean squared error loss and Adam optimizer
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

# Train the model
history = model.fit(c, f, epochs=500, verbose=0)
print("Training completed")

# Predict temperature in Fahrenheit for 100 degrees Celsius
print("Predicted temperature for 100°C:", model.predict([100]))

# Get the trained weights of the model
print("Trained weights:", model.get_weights())

# Plot the loss history during training
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()
