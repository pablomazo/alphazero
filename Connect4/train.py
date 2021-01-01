import torch

from collections import namedtuple
from random import random
import sys
sys.path.append('../')

from MCTS import MCTS
from Node import Node
from Game import Connect4
from ReplayMemory import ReplayMemory
from DNN import DNN
from Trainer import Trainer
import Tournament

torch.set_num_threads(1)

# Train loop:
def self_play_game():
    # Play games to fill replay memory
    T = 0.5
    end = False
    player = -1

    game = Connect4()
    root = Node(game.init_state, 1, player)
    mcts = MCTS(game, dnn, ngames=NGAMES)

    history, winner = mcts.play(root, T)

    # Save play in replaymemory
    for node in history:
        policy = torch.tensor([0,0,0,0,0,0,0],dtype=torch.float)
        p = mcts.eval_policy(node, 1)
        for i, a in enumerate(game.avail_actions(node.state)):
            policy[a] = p[i]

        replay_memory.add(node.state, policy, torch.tensor([-winner*node.player], dtype=torch.float))

    return winner

# Fill replayMemory with some data to check
CAPACITY = 10000
REPLAY_START_SIZE = 100
MINIBATCH = 40
BATCH = 30
NEPOCHS = 2000
NGAMES = 150
GAMES_PER_EPISODE = 10
EPISODES = 100

TOURNAMENT_FREQ = 10


Transition = namedtuple('Transition',('state', 'policy', 'reward'))
dnn = DNN()
replay_memory = ReplayMemory(CAPACITY, REPLAY_START_SIZE)


optimizer = torch.optim.SGD(dnn.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

trainer = Trainer(NEPOCHS, MINIBATCH, BATCH)

loss_hist = []

dnn.save_checkpoint(name='best.pth')
for episode in range(1,EPISODES+1):
    # Play self games to fill memory
    for igame in range(GAMES_PER_EPISODE):
        w = self_play_game()

    # Train
    loss = trainer.train(dnn, optimizer, replay_memory)

    if episode % TOURNAMENT_FREQ == 0:
        dnn.save_checkpoint(name='checkpoint.pth')

        # Make tournament between the two models:
        nwin1, nwin2, draw = Tournament.tournament('checkpoint.pth', 'best.pth', NGAMES=100)

        if nwin1 > nwin2:
            # Current player is better:
            dnn.save_checkpoint(name='best.pth')
        else:
            # Reload previous model:
            dnn.load_state_dict(torch.load('best.pth'))

    print('Episode, Loss, replay_memory len:', episode, loss, len(replay_memory))
