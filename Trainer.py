from collections import namedtuple
import torch

# Define replay memory:
Transition = namedtuple('Transition',\
        ('state', 'policy', 'reward'))

class Trainer:
    def __init__(self, nepochs, mini_batch, batch):
        self.nepochs = nepochs
        self.mini_batch = mini_batch
        self.batch = batch

        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def loss_fn(self,z, v, policy, net_pol):
        a = self.mse(z,v)
        b = self.ce(net_pol, policy.view(-1))
        return a + b

    def train(self, model, optimizer, memory):
        if len(memory) < memory.REPLAY_START_SIZE:
            return

        # Training process is run for self.nepochs epochs.
        for epoch in range(self.nepochs):
            loss = 0

            # Take self.mini_batch mini batches:
            for ibatch in range(self.mini_batch):
                l = self.optimize(model, optimizer, memory)
                loss += l

            loss /= self.batch

            if epoch % 100 == 0:
                print(epoch, loss)
        return loss

    def optimize(self, model, optimizer, memory):
        transitions = memory.sample(self.batch)
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state).reshape([self.batch,-1])
        policy = torch.cat(batch.policy).reshape([self.batch,-1])
        z = torch.cat(batch.reward).reshape([self.batch,-1])

        net_pol, v = model(states)

        loss = self.loss_fn(z,v,policy,net_pol)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.data.item()
