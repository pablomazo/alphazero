import torch
import numpy as np

class Connect4:
    def __init__(self):
        self.NCOL = 7
        self.NROW = 6
        self.NINLINE = 4

    def play(self, state, a , player):
        if type(state) is list:
            new_state = state.copy()
        elif torch.is_tensor(state):
            new_state = state.detach().clone()

        for irow in range(self.NROW):
            if state[irow,a] == 0:
                new_state[irow, a] = player
                return new_state

    def check_end(self, state):
        # Last played pieces must be located on top of each 
        # column:
        last_played = []
        for jp in range(self.NCOL):
            for ip in range(self.NROW-1,-1,-1):
                if state[ip,jp] != 0:
                    last_played.append([ip,jp])

        for [ip, jp] in last_played:
            #--------------------------------------------------
            # Check horizontal line:
            suma = 1
            j = jp + 1
            while  j < self.NCOL and state[ip, j] == state[ip, jp]:
                suma += 1
                j += 1

            j = jp - 1
            while j >= 0 and state[ip,j] == state[ip, jp]:
                suma += 1
                j -= 1

            if suma >= self.NINLINE:
                end = True
                winner = state[ip,jp]
                return end, winner
            #--------------------------------------------------

            # Check vertical line:
            suma = 1
            i = ip + 1
            while i < self.NROW and state[i, jp] == state[ip, jp]:
                suma += 1
                i += 1

            i = ip - 1
            while  i >= 0 and state[i,jp] == state[ip, jp]:
                suma += 1
                i -= 1

            if suma >= self.NINLINE:
                end = True
                winner = state[ip,jp]
                return end, winner
            #--------------------------------------------------

            # Check  diagonal line:
            suma = 1
            i, j = ip - 1, jp + 1
            while i >= 0 and j < self.NCOL and\
                  state[i,j] == state[ip, jp]:
                  suma += 1
                  i -= 1
                  j += 1

            i, j = ip + 1, jp - 1
            while i < self.NROW and j >= 0  and\
                  state[i,j] == state[ip, jp]:
                  suma += 1
                  i += 1
                  j -= 1

            if suma >= self.NINLINE:
                end = True
                winner = state[ip,jp]
                return end, winner
            #--------------------------------------------------

            # Check  inverse diagonal line:
            suma = 1
            i, j = ip + 1, jp + 1
            while i < self.NROW and j < self.NCOL and\
                  state[i,j] == state[ip, jp]:
                  suma += 1
                  i += 1
                  j += 1

            i, j = ip - 1, jp - 1
            while i >= 0 and j >= 0 and\
                  state[i,j] == state[ip, jp]:
                  suma += 1
                  i -= 1
                  j -= 1

            if suma >= self.NINLINE:
                end = True
                winner = state[ip,jp]
                return end, winner

        # Check if there are available actions:
        if len(self.avail_actions(state)) == 0:
            end = True
            winner = 0
            return end, winner

        return False, 0

    def avail_actions(self, state):
        # Returns the available actions for a given board state.
        actions = []

        for col in range(self.NCOL):
            if (state[-1,col] == 0): actions.append(col)

        return actions

    def plot(self, state):
        # p1: x  p2: o
        for i in range(self.NROW-1,-1,-1):
            print('-----------------------------')
            out = '| '
            for j in range(0, self.NCOL):
                if state[i, j] == 1:
                    token = 'x'
                if state[i, j] == -1:
                    token = 'o'
                if state[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-----------------------------')
