import numpy as np
from keras.datasets import imdb
import json
num_words = 100000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

from keras.preprocessing import sequence
X_train = sequence.pad_sequences(X_train, maxlen = 500)
X_test = sequence.pad_sequences(X_test, maxlen = 500)

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

model = Sequential()
model.add(Embedding(num_words, 32, input_length = 500))
model.add(LSTM(units = 100))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 64, epochs = 6)

model_json = model.to_json()
with open("Model_Save/model_json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("Model_Save/model.h5")

if __name__ == '__main__':
    print(model.summary())
    scores = model.evaluate(X_test, y_test, verbose=0) 
    print("Accuracy: %.2f%%" % (scores[1]*100))








