import numpy as np


class DQN():

    def __init__(self, max_memory, discount_factor):
        self.max_memory = max_memory
        self.discount_factor = discount_factor
        self.memory = list()

    def remember(self, observation, gameover):
        self.memory.append([observation, gameover])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def batch(self, model, batch_size=16):
        lenMemory = len(self.memory)
        nb_inputs = self.memory[0][0][0].shape[1]
        nb_outputs = model.output_shape[-1]

        inputs = np.zeros((min(batch_size, lenMemory), nb_inputs))
        targets = np.zeros((min(batch_size, lenMemory), nb_outputs))

        for i, inx in enumerate(np.random.randint(0, lenMemory, size = min(batch_size, lenMemory))):
            current_state, action, reward, next_state = self.memory[inx][0]
            game_over = self.memory[inx][1]

            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            if game_over:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.discount_factor * np.max(model.predict(next_state)[0])

        return inputs, targets








