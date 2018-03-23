from spaces import Discrete
class DBNode():
    Nodes = []
    def __init__(self, weights, index):
        '''
        nodes - list of other dbnodes that new node is connected to
        weights - assigned factor weights for new node
        '''
        self.weights = weights
        self.index = index
        self.action_space = Discrete(3)
        #0 is left, 1 is stationary, 2 is right
        Nodes.append(self)