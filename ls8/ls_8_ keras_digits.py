import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # Mnist dataset library
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

# Load MNIST dataset and split into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Standardize the input data by scaling pixel values to the range [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# Convert labels to categorical format
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Display the first 25 images from the training dataset
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

# Create a sequential model with layers
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())  # Display the neural network structure in the console

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using training data
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

# Evaluate the model on the test data
model.evaluate(x_test, y_test_cat)

n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(np.argmax(res))

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Recognition of the entire test dataset
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

# Identify incorrect predictions
mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
y_false = x_test[~mask]

print(x_false.shape)

# Display the first 25 incorrect prediction results
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)

plt.show()
