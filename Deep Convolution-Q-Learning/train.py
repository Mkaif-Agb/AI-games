# Snake: Deep Convolutional Q-Learning - Training file

# Importing the libraries
from environment import Environment
from brain import Brain
from dqn import DQN
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from termcolor import colored

learning_rate = 0.0001
max_memory = 90000
gamma = 0.9
batch_size = 32
nLaststates = 4
epsilon = .5
epsilon_decay_rate = 0.0002
min_epsilon = 0.05

filepath_save = 'model3.h5'

env = Environment(0)
brain = Brain((env.nColumns, env.nRows, nLaststates), learning_rate=learning_rate)
#model = brain.model
model = load_model(filepath_save)
print(model.summary())
dqn = DQN(max_memory=max_memory, discount_factor=gamma)

def reset_states():
    current_state = np.zeros((1, env.nColumns, env.nRows, nLaststates))

    for i in range(nLaststates):
        current_state[0, :, :, i] = env.screenMap

    return current_state, current_state


# Training
epoch = 0
nCollected = 0
maxnCollected = 0
totalnCollected = 0
scores = list()

while True:
    epoch += 1
    env.reset()
    currentState, nextState = reset_states()
    game_over = False
    testing = True
    rewards = list()
    while not game_over :

        if np.random.rand() <= epsilon and not testing:
            action = np.random.randint(0, 4)
        else:
            q_values = model.predict(currentState)[0]
            action = np.argmax(q_values)

        frame, reward , game_over = env.step(action=action)
        rewards.append(reward)
        frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
        nextState = np.append(nextState, frame, axis=3)
        nextState = np.delete(nextState, 0, axis=3)

        dqn.remember([currentState, action, reward, nextState], game_over)
        inputs, targets = dqn.batch(model=model, batch_size=batch_size)
        model.train_on_batch(inputs, targets)


        if env.collected:
            nCollected += 1
        currentState = nextState

    epsilon -= epsilon_decay_rate
    epsilon = max(epsilon, min_epsilon)

    if nCollected > maxnCollected and nCollected> 2:
        print('Trained Model Saved')
        model.save(filepath=filepath_save)
        maxnCollected = nCollected

    totalnCollected += nCollected
    nCollected = 0

    if epoch % 100 == 0 and epoch != 0:
        scores.append(totalnCollected/100)
        # totalnCollected = 0
        # plt.plot(scores)
        # plt.xlabel('Epochs')
        # plt.ylabel('Avg Collected')
        # plt.title("Snake Game Rewards")
        # plt.show()
    if np.sum(rewards) > 0:
        col = 'green'
    else:
        col='red'
    print('Epoch {} Epsilon {:.5f} Apples_Collected {} Reward'.format(epoch, epsilon, maxnCollected), end=' ')
    print(colored('{}'.format(np.sum(rewards)), col))














