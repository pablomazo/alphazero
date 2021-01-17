import numpy as np
import sys
sys.path.append('../')

import torch
from Game import Game
from DNN import DNN
from MCTS import MCTS
from Node import Node

def tournament(agent1, agent2, NGAMES=100, mcts_games=50):
    T = 0
    dnn1 = DNN()
    if agent1 != None:
        dnn1.load_state_dict(torch.load(agent1))

    dnn2 = DNN()
    if agent2 != None:
        dnn2.load_state_dict(torch.load(agent2))

    dnn1_wins = 0
    dnn2_wins = 0
    draw = 0

    game = Game()

    for igame in range(NGAMES):
        if igame % 10 == 0: print('GAME: {} in tournament'.format(igame))
        p1 = np.random.choice(2)
        if p1 == 0:
            player1 = dnn1
            player2 = dnn2
        elif p1 == 1:
            player1 = dnn2
            player2 = dnn1

        mcts1 = MCTS(game, player1, ngames=mcts_games)
        mcts2 = MCTS(game, player2, ngames=mcts_games)
        end = False
        player = -1
        node1 = Node(game.init_state, 1, player)
        node2 = Node(game.init_state, 1, player)

        while not end:
            if node1.player == 1:
                mcts1.explore(node1)
                a, p = mcts1.select_action(node1, T)

                if node2.children == []:
                    mcts2.explore(node2)

            elif node1.player == -1:
                mcts2.explore(node2)
                a, p = mcts2.select_action(node2, T)

                if node1.children == []:
                    mcts1.explore(node1)

            # To use previous searches on nodes.
            node1 = node1.children[a]
            node2 = node2.children[a]
            end, w = node1.end, node1.v

        if w == 1:
            if p1 == 0:
                dnn1_wins += 1
            else:
                dnn2_wins += 1
        elif w == -1:
            if p1 == 0:
                dnn2_wins += 1
            else:
                dnn1_wins += 1
        else:
            draw += 1

    dnn1_wins /= NGAMES
    dnn2_wins /= NGAMES
    draw /= NGAMES
    print('Player1 win rate:', dnn1_wins)
    print('Player2 win rate:', dnn2_wins)
    print('Draw rate:', draw)
    return dnn1_wins, dnn2_wins, draw
