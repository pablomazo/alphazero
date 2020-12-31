import numpy as np
import sys
sys.path.append('../')

import torch
from Game import Connect4
from DNN import DNN
from MCTS import MCTS
from Node import Node


dnn1 = DNN()
dnn1.load_state_dict(torch.load('best.pth'))

player1_wins = 0
player2_wins = 0
draw = 0

game = Connect4()

p1 = 10
while p1 != 0 and p1 != 1:
    p1 = int(input('0: Máquina juega primero; 1: Humano juega primero'))
player1 = dnn1

mcts1 = MCTS(game, player1, ngames=50)
end = False
player = -1
node = Node(game.init_state, 1, player)

game.plot(node.state)

while not end:
    if p1 == 0:
        if player == -1:
            mcts1.explore(node)
            a = mcts1.select_action(node, 0.01)
        elif player == 1:
            a = int(input())
            if node.children == []:
                mcts1.explore(node)

    if p1 == 1:
        if player == 1:
            mcts1.explore(node)
            a = mcts1.select_action(node, 0.01)
        elif player == -1:
            a = int(input())
            if node.children == []:
                mcts1.explore(node)


    print(len(node.children), a)
    node = node.children[a]
    player = node.player
    game.plot(node.state)

    end, w = game.check_end(node.state)

    # Remove childrens:
    node.children = []

print(w)
