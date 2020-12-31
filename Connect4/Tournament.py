import numpy as np
import sys
sys.path.append('../')

import torch
from Game import Connect4
from DNN import DNN
from MCTS import MCTS
from Node import Node

def tournament(agent1, agent2, NGAMES=100):

    dnn1 = DNN()
    dnn1.load_state_dict(torch.load(agent1))

    dnn2 = DNN()
    dnn2.load_state_dict(torch.load(agent2))

    dnn1_wins = 0
    dnn2_wins = 0
    draw = 0

    game = Connect4()

    for igame in range(NGAMES):
        p1 = np.random.choice(2)
        if p1 == 0:
            player1 = dnn1
            player2 = dnn2
        elif p1 == 1:
            player1 = dnn2
            player2 = dnn1

        mcts1 = MCTS(game, player1, ngames=150)
        mcts2 = MCTS(game, player2, ngames=150)
        end = False
        player = -1
        node1 = Node(game.init_state, 1, player)
        node2 = Node(game.init_state, 1, player)

        while not end:
            if node1.player == 1:
                mcts1.explore(node1)
                a = mcts1.select_action(node1, 0.01)

                if node2.children == []:
                    mcts2.explore(node2)
            elif node1.player == -1:
                mcts2.explore(node2)
                a = mcts2.select_action(node2, 0.01)

                if node1.children == []:
                    mcts1.explore(node1)

            # To use previous searches on nodes.
            node1 = node1.children[a]
            node2 = node2.children[a]
            end, w = game.check_end(node1.state)

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

    print('Player1 win rate:', dnn1_wins / NGAMES)
    print('Player2 win rate:', dnn2_wins / NGAMES)
    print('Draw rate:', draw / NGAMES)
    dnn1_wins /= NGAMES
    dnn2_wins /= NGAMES
    draw /= NGAMES
    return dnn1_wins, dnn2_wins, draw
