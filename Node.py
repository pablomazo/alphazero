class Node:
    '''
    Node class:
        Input:
            - state: State of the node.
            - P: Prior probability of node.
        Atributes:
            - Q: Mean value of the next state.
            - W: Total value of the next state.
            - N: Number of times the node was visited.
    '''
    def __init__(self, state, P):
        self.state = state
        self.P = P
        self.Q = 0e0
        self.W = 0e0
        self.N = 0
        self.children = []
