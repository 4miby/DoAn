import random
import numpy as np
class MemoryBuffer:
    def __init__(self, e_max=15000, e_min=100):
        self._max = e_max  # Số lượng kinh nghiệm tối đa
        self._min = e_min  # Số lượng kinh nghiệm tối thiểu
        self.exp = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}

    def get_max(self):
        return self._max

    def get_min(self):
        return self._min

    def get_num(self):
        return len(self.exp['state'])

    def sample(self, batch_size=64):
        idx = np.random.choice(self.get_num(), size=batch_size, replace=False)
        states = [self.exp['state'][i] for i in idx]
        next_states = [self.exp['next_state'][i] for i in idx]

        # Tìm độ dài tối đa của các trạng thái
        max_length = max(max(len(state) for state in states), max(len(state) for state in next_states))

        # Pad states và next_states để có cùng độ dài
        states_padded = np.array([np.pad(state, (0, max_length - len(state)), 'constant') for state in states])
        next_states_padded = np.array([np.pad(state, (0, max_length - len(state)), 'constant') for state in next_states])

        action = [self.exp['action'][i] for i in idx]
        reward = [self.exp['reward'][i] for i in idx]
        done = [self.exp['done'][i] for i in idx]
        return states_padded, action, reward, next_states_padded, done

    def add(self, state, action, reward, next_state, done):
        if self.get_num() > self.get_max():
            del self.exp['state'][0]
            del self.exp['action'][0]
            del self.exp['reward'][0]
            del self.exp['next_state'][0]
            del self.exp['done'][0]
        self.exp['state'].append(state)
        self.exp['action'].append(action)
        self.exp['reward'].append(reward)
        self.exp['next_state'].append(next_state)
        self.exp['done'].append(done)
