import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Brain():

    def __init__(self, nb_inputs, nb_outputs, learning_rate=0.001):
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.learning_rate = learning_rate

        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_shape=(self.nb_inputs,)))
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dense(units=16, activation='relu'))
        self.model.add(Dense(units=nb_outputs))

        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss='mean_squared_error', )






