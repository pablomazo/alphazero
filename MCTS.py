import math
import torch
from random import random, randint

from Node import Node

class MCTS:
    '''
    MCTS class.

    Input:
        - game: Game object, with set of rules, end conditions, ...
        - state: Root node objetc.
        - NN: deep neural network with policy and value nets.
        - ngames: Number of games to sample best action.
    '''
    def __init__(self, game, state, NN, ngames=10):
        self.root = state
        self.game = game
        self.NN = NN
        self.ngames = ngames
        self.player = self.root.player

    def select(self, root):
        node = root
        history = []
        end = False

        while not end:
            if node.children != []:
                actions = self.game.avail_actions(node.state)
                node = self.choose(node.children)
                node.N += 1
                history.append(node)

                end, v = self.game.check_end(node.state)
                self.player *= -1

            else:
                # Expand node.
                actions = self.game.avail_actions(node.state)
                v = self.expand(node, actions)

                end = True

        # Backup edges.
        self.backup(history, v)

        return history

    def backup(self, history, v):
        for node in reversed(history):
            node.W += v if self.root.player == (-node.player) else -v
            node.Q = node.W / node.N

    def choose(self, children):
        Q = [child.Q for child in children]
        U = self.Uval(children)

        total = [sum(x) for x in zip(Q,U)]

        a_idx = total.index(max(total))
        return children[a_idx]

    def expand(self, node, actions):
        with torch.no_grad():
            p, v = self.NN(node.state)

        for a in actions:
            new_state = self.game.play(node.state, a, self.player)
            node.children.append(Node(new_state, p[a], self.player))

        return v

    def play(self, node, temperature):
        policy = self.eval_policy(node, temperature)

        if temperature == 1e0:
            a_idx = policy.index(max(policy))

        else:
            Nact = len(node.children)
            a_idx = self.sample(policy, Nact)

        return a_idx

    def eval_policy(self, node, temperature):
        N = [child.N**(1e0/temperature) for child in node.children]
        Nall = sum(N)
        policy = [float(N_a / Nall) for N_a in N]
        return policy

    def sample(self, policy, Nact):

        while True:
            a_idx = randint(0,Nact-1)
            v = random()

            if v < policy[a_idx]:
                return a_idx

    def Uval(self, children):
        cpuct = 1e0
        U = []
        Nall = sum(child.N for child in children)
        Nall = math.sqrt(Nall)

        for child in children:
            aux = cpuct * child.P * Nall / (1e0 + child.N)
            U.append(aux)

        return U

    def explore(self, state):
        for igame in range(self.ngames):
            self.player = state.player * -1
            _ = self.select(state)
