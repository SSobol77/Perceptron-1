import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

# Read and preprocess the text data
with open('train_data_true', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')  # Remove the first invisible character
    text = re.sub(r'[^А-я ]', '', text)  # Replace all characters except Cyrillic letters with empty characters

# Tokenize the text as a sequence of characters
num_characters = 34  # 33 letters + space
tokenizer = Tokenizer(num_words=num_characters, char_level=True)  # Tokenize at the character level
tokenizer.fit_on_texts([text])  # Generate tokens based on frequency in the text
print(tokenizer.word_index)

inp_chars = 6
data = tokenizer.texts_to_matrix(text)  # Convert the original text to one-hot encoding array
n = data.shape[0] - inp_chars  # Since we predict based on three characters - the fourth one

X = np.array([data[i:i + inp_chars, :] for i in range(n)])
Y = data[inp_chars:]  # Predict the next character

print(data.shape)

# Build and compile the RNN model
model = Sequential()
model.add(Input((inp_chars, num_characters)))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(num_characters, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
history = model.fit(X, Y, batch_size=32, epochs=100)

# Function to build a phrase using the trained model
def buildPhrase(inp_str, str_len=50):
    for i in range(str_len):
        x = []
        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_str[j]))

        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters)

        pred = model.predict(inp)
        d = tokenizer.index_word[pred.argmax(axis=1)[0]]

        inp_str += d

    return inp_str

# Generate a phrase using the model
res = buildPhrase("утренн")
print(res)
