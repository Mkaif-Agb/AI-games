# Libraries
import numpy as np
from environment import Environment


gamma = 0.9
alpha = 0.75
epochs = 20000

# Initializing rewards and Q_table
env = Environment()
rewards = env.rewardBoard
Q = rewards.copy()

# list of all the possible states that can be reached (Removing the brick wall)
# possible state is the rows indexed 0

possible_states = list()
for i in range(rewards.shape[0]):
    if sum(abs(rewards[i])) != 0:
        possible_states.append(i)

# Finding the maximum Q-value and its respective index

def maximum_q(qvalues):
    inx = 0
    max_Q = -np.inf
    for i in range(len(qvalues)):
        if qvalues[i] > max_Q and qvalues[i] != 0:
            max_Q = qvalues[i]
            inx = i
    return inx, max_Q


for epoch in range(epochs):
    print('\r Epoch: ' + str(epoch+1), end='' )
    starting_position = np.random.choice(possible_states)

    # Possible actions are the columns indexed 1
    possible_actions = list()
    for i in range(rewards.shape[1]):
        if rewards[starting_position][i] != 0:
            possible_actions.append(i)

    action = np.random.choice(possible_actions)
    reward = rewards[starting_position][action]

    _, max_Q = maximum_q(Q[action])
    TD = reward + gamma * max_Q - Q[starting_position][action]
    Q[starting_position][action] = Q[starting_position][action] + alpha * TD


current_pos = env.startingPos
while True:
    action, _ = maximum_q(Q[current_pos])
    env.movePlayer(action)
    current_pos = action

