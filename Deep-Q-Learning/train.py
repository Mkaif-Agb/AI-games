
from dqn import DQN
from brain import Brain
import gym
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
memory_max = 50000
gamma = 0.9
batch_size = 32
epsilon = 1.
epsilon_decay = 0.995


env = gym.make('MountainCar-v0')
brain = Brain(2, 3, learning_rate=learning_rate)
model = brain.model
dqn = DQN(max_memory=memory_max, discount_factor=gamma)

epoch = 0
currentState = np.zeros((1, 2))
nextState = currentState
total_reward = 0
rewards = list()
while True:
    epoch += 1

    env.reset()
    currentState = np.zeros((1,2))
    nextState = currentState
    game_over = False
    while not game_over:

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 3)
        else:
            q_values = model.predict(currentState)[0]
            action = np.argmax(q_values)  # index

        nextState[0], reward, game_over, _ = env.step(action)
        env.render()

        total_reward += reward
        dqn.remember([currentState, action, reward, nextState],game_over)
        inputs, targets = dqn.batch(model= model, batch_size= batch_size)
        model.train_on_batch(inputs, targets)

        currentState = nextState

    epsilon *= epsilon_decay
    print('Epoch {} Epsilon {:.5f} Reward {:.2f}'.format(epoch, epsilon, total_reward))
    rewards.append(total_reward)
    total_reward = 0

    if epoch % 100 == 0:
        plt.plot(rewards)
        plt.xlabel('Epochs')
        plt.ylabel('Rewards')
        plt.title('Rewards Gained')
        plt.show()
        model.save('Mountain-Car-v0.h5')


env.close()


