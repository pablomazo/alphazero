import torch

import numpy as np
from collections import namedtuple
from random import random
import sys
from argparse import ArgumentParser
sys.path.append('../')

from MCTS import MCTS
from Node import Node
from Game import Game
from ReplayMemory import ReplayMemory
from DNN import DNN
from Trainer import Trainer
import Tournament

torch.set_num_threads(1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes in training process.")
    parser.add_argument("--ngames", type=int, default=200, help="Number of self-play games in each MCTS.")
    parser.add_argument("--games_per_episode", type=int, default=100, help="Number of self-play games before each training episode.")
    parser.add_argument("--tournament_freq", type=int, default=1, help="Frequency, in episodes, to make tournament between current player and best player.")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size.")
    parser.add_argument("--nbatch", type=int, default=40, help="Number of batches at each epoch.")
    parser.add_argument("--nepochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--capacity", type=int, default=5000, help="Maximum number of datapoints in replaymemory.")
    parser.add_argument("--replay_start_size", type=int, default=100, help="Minimum number of datapoint to start training.")
    args = parser.parse_args()

# Play games to fill replay memory
def self_play_game(best_model):
    T = 1.0
    end = False
    player = -1

    game = Game()
    root = Node(game.init_state, 1, player)
    mcts = MCTS(game, best_model, ngames=args.ngames)

    g, winner = mcts.play(root, T)

    # Save play in replaymemory
    for node, p in g:
        avail = game.avail_actions(node)
        a = torch.zeros(game.nactions)
        for i, act in enumerate(avail):
            a[act] = p[i]
        w = torch.tensor([winner*node.player], dtype=torch.float)

        replay_memory.add(node.state, a, w)

    return winner

if __name__ == "__main__":
    print('Train settings:')
    print(args)

    Transition = namedtuple('Transition',('state', 'policy', 'reward'))
    dnn_best = DNN()
    dnn = DNN()
    dnn.load_state_dict(dnn_best.state_dict())

    replay_memory = ReplayMemory(args.capacity, args.replay_start_size)

    optimizer = torch.optim.AdamW(dnn.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = Trainer(args.nepochs, args.nbatch, args.batch_size)

    dnn_best.save_checkpoint(name='best.pth')
    for episode in range(1,args.episodes+1):
        # Play self games to fill memory
        for igame in range(args.games_per_episode):
            if igame % 10 == 0:
                print('Starting {} self play game'.format(igame))
            # Self plays use the best current policy.
            w = self_play_game(dnn_best)

        # Train
        loss = trainer.train(dnn, optimizer, replay_memory)

        if episode % args.tournament_freq == 0:
            dnn.save_checkpoint(name='checkpoint.pth')

            # Make tournament between the two models:
            nwin1, nwin2, draw = Tournament.tournament('checkpoint.pth', 'best.pth', NGAMES=100, mcts_games=args.ngames)

            if nwin1 >= 0.55:
                # Current player is better:
                dnn_best.load_state_dict(dnn.state_dict())
                dnn.save_checkpoint(name='best.pth')
                dnn.save_checkpoint(name='checkpoint_{}.pth'.format(episode))

        print('Episode, Loss, replay_memory len:', episode, loss, len(replay_memory))
