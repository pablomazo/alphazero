import math
import torch
from random import random, randint
import numpy as np

from Node import Node

class MCTS:
    '''
    MCTS class.

    Input:
        - game: Game object, with set of rules, end conditions, ...
        - NN: deep neural network with policy and value nets.
        - ngames: Number of games to sample best action.
    '''
    def __init__(self, game, NN, ngames=10):
        self.root = None
        self.game = game
        self.NN = NN
        self.ngames = ngames

    def simulation(self, root):
        node = root
        history = []
        end = False

        while not end:
            actions = self.game.avail_actions(node.state)
            if node.children != []:
                node = self.choose(node.children)
                history.append(node)

                end, v = self.game.check_end(node.state)
                self.player *= -1

            else:
                # Expand node.
                v = self.expand(node, actions)

                end = True

        # Backup edges.
        self.backup(history, v)

        return history

    def backup(self, history, v):
        for node in reversed(history):
            node.N += 1
            node.W += v * node.player
            node.Q = node.W / node.N

    def choose(self, children):
        Q = [child.Q for child in children]
        U = self.Uval(children)

        total = [sum(x) for x in zip(Q,U)]

        a_idx = total.index(max(total))
        return children[a_idx]

    def expand(self, node, actions):
        p, v = self.NN.eval(node.state)

        for a in actions:
            new_state = self.game.play(node.state, a, self.player)
            node.children.append(Node(new_state, p[a], self.player))

        return v

    def select_action(self, node, temperature):
        policy = self.eval_policy(node, temperature)

        a_idx = self.sample(policy)

        return a_idx

    def eval_policy(self, node, temperature):
        N = [child.N**(1e0/temperature) for child in node.children]
        Nall = sum(N)
        policy = [float(N_a / Nall) for N_a in N]
        return policy

    def sample(self, policy):
        Nact = len(policy)

        a_idx = np.random.choice(Nact, p=policy)
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

    def add_dirichlet(self, actions):
        root_dirichlet_alpha = 0.3
        root_exploration_fraction = 0.25
        # Add Dirichlet noise.
        noise = np.random.dirichlet([root_dirichlet_alpha] * len(actions))
        frac = root_exploration_fraction
        for a, n in enumerate(noise):
            self.root.children[a].P = self.root.children[a].P * (1 - frac) + n * frac


    def explore(self, root):
        self.root = root
        self.player = self.root.player

        # Expand root node if no children.
        actions = self.game.avail_actions(self.root.state)
        if self.root.children == []:
            self.player = self.root.player * -1
            _ = self.expand(self.root, actions)

        # Add noise to children in root node.
        self.add_dirichlet(actions)

        for igame in range(self.ngames):
            self.player = self.root.player * -1
            _ = self.simulation(self.root)

    def play(self, node, iniT):
        end = False
        player = -node.player
        history = []

        while not end:
            T = np.amax([iniT, 0.1])
            history.append(node)
            self.explore(node)
            a = self.select_action(node, T)
            node = node.children[a]
            end, winner = self.game.check_end(node.state)

            # Temperature is reduced to make deterministic moves as game advances.
            iniT -= 0.2

        return history, winner
