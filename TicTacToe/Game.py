import torch
import numpy as np

class Game:
    def __init__(self):
        self.init_state = torch.zeros(9, dtype=torch.float)
        self.nactions = 9

    def play(self, node, a):
        new_state = node.state.detach().clone()
        new_state[a] = node.player * -1
        return new_state

    def check_end(self, node):
        win_arrays = [[0,1,2], [3,4,5], [6,7,8], # horizontal
                      [0,3,6], [1,4,7], [2,5,8], # vertical
                      [0,4,8], [2,4,6]]          # diagonal

        state = node.state
        # If there are less than 5 pieces there is no possible winner.
        if torch.sum(torch.abs(state)) < 5:
            return False, 0

        # Check winning lines:
        for h in win_arrays:
            v = torch.sum(state[h])

            if torch.abs(v) == 3:
                end = True
                winner = 1 if v > 0 else -1
                return end, winner

        # Check if there are available positions.
        if 0 not in state:
            end = True
            winner = 0
            return end, winner

        # The game did not finish.
        return False, 0

    def avail_actions(self, node):
        # Returns the available actions for a given board state..
        actions = []

        state = node.state
        # Each action is given as a vector with a one in the free position.
        for a, pos in enumerate(state):
            if pos == 0:
                actions.append(a)

        return actions

    def plot(self, state):
        # p1: x  p2: o
        for i in range(0, 3):
            print('-------------')
            out = '| '
            for j in range(0, 3):
                if state[3*i+j] == 1:
                    token = 'x'
                if state[3*i+j] == -1:
                    token = 'o'
                if state[3*i+j] == 0:
                    token = str(3*i+j)
                out += token + ' | '
            print(out)
        print('-------------')
