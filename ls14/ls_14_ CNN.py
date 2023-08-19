import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # Mnist dataset library
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

# Load MNIST dataset and split into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Standardize the input data by scaling pixel values to the range [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# Convert labels to categorical format
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Expand dimensions to account for convolutional input shape
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Print the shape of the training data after expansion
print( x_train.shape )

# Create a sequential model with convolutional and pooling layers
model = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# Train the model using training data with validation split
his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

# Evaluate the model's performance on the test data
model.evaluate(x_test, y_test_cat)
