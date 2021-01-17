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
            if node.avail_actions == None:
                actions = self.game.avail_actions(node)
                node.avail_actions = actions
            else:
                actions = node.avail_actions

            if node.children != []:
                node = self.choose(node.children)
                history.append(node)

                if node.end == None:
                    end, v = self.game.check_end(node)
                    node.end = end
                    node.v = v
                else:
                    end = node.end
                    v = node.v

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
        Q = np.array([child.Q for child in children])
        U = self.Uval(children)

        total = Q + U

        a_idx = np.argmax(total)
        return children[a_idx]

    def expand(self, node, actions):
        p, v = self.NN.eval(node.state)
        player = node.player * -1

        for a in actions:
            new_state = self.game.play(node, a)
            node.children.append(Node(new_state * player, p[a], player))
            node.children[-1].last_action = a

        return v * node.player

    def select_action(self, node, T):
        policy = self.eval_policy(node, T)
        Nact = policy.shape[0]

        a_idx = np.random.choice(Nact, p=policy)

        return a_idx, policy

    def eval_policy(self, node, T):
        if T == 0:
            N = np.array([child.N for child in node.children])
            p = np.argmax(N)
            policy = np.zeros(N.shape[0])
            policy[p] = 1e0
        else:
            N = np.array([child.N**(1e0/T) for child in node.children])
            Nall = np.sum(N)
            policy = N / Nall

        return policy

    def Uval(self, children):
        cpuct = 1e0
        P = []
        N = []
        for child in children:
            P.append(child.P)
            N.append(child.N)

        P = np.array(P)
        N = np.array(N)
        Nall = np.sum(N)
        Nall = np.sqrt(Nall)

        U = cpuct * P * Nall / (1e0 + N)

        return U

    def add_dirichlet(self, actions):
        root_dirichlet_alpha = 0.03
        root_exploration_fraction = 0.25
        # Add Dirichlet noise.
        noise = np.random.dirichlet([root_dirichlet_alpha] * len(actions))
        frac = root_exploration_fraction
        for a, n in enumerate(noise):
            self.root.children[a].P = self.root.children[a].P * (1 - frac) + n * frac


    def explore(self, root):
        self.root = root

        # Expand root node if no children.
        if root.avail_actions == None:
            actions = self.game.avail_actions(root)
            root.avail_actions = actions
        else:
            actions = root.avail_actions

        if self.root.children == []:
            _ = self.expand(self.root, actions)

        # Add noise to children in root node.
        self.add_dirichlet(actions)

        for igame in range(self.ngames):
            _ = self.simulation(self.root)

    def play(self, node, iniT):
        end = False
        game = []
        move = 0

        while not end:
            T = iniT
            self.explore(node)
            a, p = self.select_action(node, T)
            game.append([node, p])
            node = node.children[a]
            end, winner = node.end , node.v

            move += 1

        return game, winner
