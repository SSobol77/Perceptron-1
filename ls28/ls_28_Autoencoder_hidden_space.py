import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input

# Load the MNIST dataset and split into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Standardize input data to the range [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# Reshape the data to match the input shape of the model
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Define the input layer for the autoencoder
input_img = Input(shape=(28, 28, 1))

# Build the encoder layers
x = Flatten()(input_img)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
encoded = Dense(2, activation='linear')(x)

# Define an input layer for the decoder
input_enc = Input(shape=(2,))

# Build the decoder layers
d = Dense(64, activation='relu')(input_enc)
d = Dense(28*28, activation='sigmoid')(d)
decoded = Reshape((28, 28, 1))(d)

# Create separate encoder and decoder models
encoder = keras.Model(input_img, encoded, name="encoder")
decoder = keras.Model(input_enc, decoded, name="decoder")

# Create the autoencoder model by combining encoder and decoder
autoencoder = keras.Model(input_img, decoder(encoder(input_img)), name="autoencoder")
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=64,
                shuffle=True)

# Encode test data and visualize the encoded space
encoded_images = encoder.predict(x_test)
scatter_plot = plt.scatter(encoded_images[:, 0], encoded_images[:, 1])

# Decode a specific point in the encoded space and visualize the reconstructed image
decoded_image = decoder.predict(np.expand_dims([50, 250], axis=0))
plt.imshow(decoded_image.squeeze(), cmap='gray')

# Show the plots
plt.show()
