# Snake: Deep Convolutional Q-Learning - Brain file

# Importing the libraries

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import RMSprop

class Brain():

    def __init__(self, input_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.nb_outputs = 4

        self.model = Sequential()
        self.model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64 ,(3,3), activation='relu'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(96 ,(2,2), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512 , activation='relu'))
        self.model.add(Dense(128 , activation='relu'))
        self.model.add(Dense(self.nb_outputs))

        self.model.compile(optimizer=RMSprop(lr=self.learning_rate), loss='mse')

    def load_model(self, filepath):
        self.model = load_model(filepath=filepath)
        return self.model



