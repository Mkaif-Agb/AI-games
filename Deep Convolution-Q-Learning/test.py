# Snake: Deep Convolutional Q-Learning - Testing file

# Importing the libraries
from environment import Environment
from brain import Brain
import numpy as np

waittime = 75
nLaststates = 4
filepathtoopen = 'model3.h5'

env = Environment(waittime)
brain = Brain((env.nColumns, env.nRows, nLaststates))
model = brain.load_model(filepath=filepathtoopen)


def reset_states():
    current_state = np.zeros((1, env.nColumns, env.nRows, nLaststates))

    for i in range(nLaststates):
        current_state[0, :, :, i] = env.screenMap

    return current_state, current_state


while True:
    ncollected = 0
    env.reset()
    currentState, nextState = reset_states()
    game_over = False
    while not game_over:
        qvalues = model.predict(currentState)[0]
        action = np.argmax(qvalues)
        frame, reward, game_over = env.step(action=action)
        frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
        nextState = np.append(nextState, frame, axis=3)
        nextState = np.delete(nextState, 0, axis=3)
        currentState = nextState
        if env.collected:
            ncollected += 1

    print("The number of apples collected are {}".format(ncollected))