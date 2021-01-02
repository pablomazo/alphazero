from collections import namedtuple
import random
import sys

# Define replay memory:
Transition = namedtuple('Transition',\
        ('state', 'policy', 'reward'))

class ReplayMemory:
    def __init__(self, capacity, start_size):
      self.capacity = capacity
      self.memory = []
      self.position = 0
      self.REPLAY_START_SIZE = start_size

      if self.REPLAY_START_SIZE > self.capacity:
          sys.exit('REPLAY_START_SIZE must be lower than capacity of replay memory')


    def add(self, *args):
        if self.__len__() < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch):
      return random.sample(self.memory, batch)

    def __len__(self):
      return len(self.memory)

