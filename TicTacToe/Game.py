import numpy as np

class TicTacToe_Board:
    def play(self, state, a, player):
        new_state = state.copy()
        new_state[a] = player
        return new_state

    def check_end(self, state):
        win_arrays = [[0,1,2], [3,4,5], [6,7,8], # horizontal
                      [0,3,6], [1,4,7], [2,5,8], # vertical
                      [0,4,8], [2,4,6]]          # diagonal

        # Check winning lines:
        for h in win_arrays:
            v = sum(state[i] for i in h)

            if abs(v) == 3:
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

    def avail_actions(self, state):
        # Returns the available actions for a given board state..
        actions = []

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
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')
