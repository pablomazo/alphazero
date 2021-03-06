import numpy as np
import sys
sys.path.append('../')

import torch
from Game import Game
from DNN import DNN
from MCTS import MCTS
from Node import Node


dnn1 = DNN()
dnn1.load_state_dict(torch.load('best.pth'))

T = 0

game = Game()

p1 = 10
while p1 != 0 and p1 != 1:
    p1 = int(input('0: Máquina juega primero; 1: Humano juega primero '))
player1 = dnn1

mcts1 = MCTS(game, player1, ngames=200)
end = False
player = -1
node = Node(game.init_state, 1, player)

game.plot(node)

while not end:
    if p1 == 0:
        if player == -1:
            mcts1.explore(node)
            a, p = mcts1.select_action(node, T)
        elif player == 1:
            avail = game.avail_actions(node)
            a = None
            while a not in avail:
                a = int(input('Your move: '))
            a = avail.index(a)
            if node.children == []:
                print('Explore')
                mcts1.explore(node)

    if p1 == 1:
        if player == 1:
            mcts1.explore(node)
            a, p = mcts1.select_action(node, T)
        elif player == -1:
            avail = game.avail_actions(node)
            a = None
            while a not in avail:
                a = int(input('Your move: '))
            a = avail.index(a)
            if node.children == []:
                print('Explore')
                mcts1.explore(node)


    print('Action', a)
    print('Count', [child.N for child in node.children])
    print('Value', [child.Q for child in node.children])
    print('Prior', [child.P for child in node.children])
    print('Total', [child.Q + child.P for child in node.children])
    node = node.children[a]
    print('New node')
    print('Count', [child.N for child in node.children])
    print('Value', [child.Q for child in node.children])
    print('Prior', [child.P for child in node.children])
    print('Total', [child.Q + child.P for child in node.children])
    player = node.player
    game.plot(node)

    end, w = game.check_end(node)

print(w)
