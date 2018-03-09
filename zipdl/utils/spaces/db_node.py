class DBNode:
    def __init__(self, weights, nodes=None):
        '''
        nodes - list of other dbnodes that new node is connected to
        weights - assigned factor weights for new node
        '''
        self.weights = weights
        if not nodes:
            self.action_space = [self]
        else:
            self.action_space = nodes #possible nodes to move to
    def add(self, *node):
        '''
        Add a possible action (ie a state to move to)
        '''
        self.action_space.extend(*node)