import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # Mnist dataset library
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Load MNIST dataset and split into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Standardize the input data by scaling pixel values to the range [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# Convert labels to categorical format
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Define a limit for the training data
limit = 5000

# Extract a subset of training data and its corresponding labels
x_train_data = x_train[:limit]
y_train_data = y_train_cat[:limit]

# Extract a subset of validation data and its corresponding labels
x_valid = x_train[limit:limit*2]
y_valid = y_train_cat[limit:limit*2]

# Create a sequential model with layers, including dropout
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(300, activation='relu'),
    Dropout(0.8),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# Train the model using training and validation data
his = model.fit(x_train_data, y_train_data, epochs=50, batch_size=32, validation_data=(x_valid, y_valid))

# Plot the training and validation loss over epochs
plt.plot(his.history['loss'], label='Training Loss')
plt.plot(his.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
