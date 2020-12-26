from collections import namedtuple
import random

# Define replay memory:
Transition = namedtuple('Transition',\
        ('state', 'policy', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
      self.capacity = capacity
      self.memory = []
      self.position = 0

    def add(self, *args):
        if self.__len__() < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch):
      return random.sample(self.memory, batch)

    def __len__(self):
      return len(self.memory)

