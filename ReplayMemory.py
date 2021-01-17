from collections import namedtuple
import random
import sys
import torch

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


    def deduplicate(self):
        sbatch = len(self.memory)

        transitions = self.sample(sbatch)
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state).reshape([sbatch,-1])
        policy = torch.cat(batch.policy).reshape([sbatch,-1])
        value = torch.cat(batch.reward).reshape([sbatch,-1])

        # Find duplicates:
        ustates, inv, c = torch.unique(states,dim=0, return_inverse=True, return_counts=True)

        # Average duplicates:
        upolicy = torch.zeros(ustates.size(0), policy.size(1))
        uvalue = torch.zeros(ustates.size(0))

        for i, elem in enumerate(inv):
            upolicy[elem] += policy[i]
            uvalue[elem] += value[i].item()

        # Store data in memory:
        self.position = 0
        self.memory = []

        for idata in range(len(upolicy)):
            self.add(ustates[idata],
                     upolicy[idata] / c[idata],
                     torch.tensor([uvalue[idata] / c[idata]]))
