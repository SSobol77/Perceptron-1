import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from google.colab import files     # Importing necessary libraries
from io import BytesIO
from PIL import Image

# Load the VGG16 model
model = keras.applications.VGG16()

# Upload an image file
uploaded = files.upload()

# Open the uploaded image and display it
img = Image.open(BytesIO(uploaded['ex224.jpg']))
plt.imshow( img )

# Convert the image to the format expected by the VGG network
img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)

# Pass the preprocessed image through the VGG network
res = model.predict( x )

# Print the predicted class label (index with the highest probability)
print(np.argmax(res))
