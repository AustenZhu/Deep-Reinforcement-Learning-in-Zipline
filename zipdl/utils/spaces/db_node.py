class DB2Node():
    Nodes2 = []
    def __init__(self, weights, index):
        '''
        nodes - list of other dbnodes that new node is connected to
        weights - assigned factor weights for new node
        '''
        self.weights = weights
        self.index = index
        self.id = len(DB2Node.Nodes2)
        #0 is left, 1 is stationary, 2 is right
        DB2Node.Nodes2.append(self)

class DBPlusNode():
    NodesPlus = []
    def __init__(self, weights, index):
        '''
        nodes - list of other dbnodes that new node is connected to
        weights - assigned factor weights for new node
        '''
        self.weights = weights
        self.index = index
        self.id = ...
        #How to represent id of nodes? - perhaps use embedding layer
        #At current moment all nodes must have same action spaces
        #TODO: Allow nodes to have different action spaces by dropping out the 
        #neurons corresponding to invalid actions for diff action spaces
        Nodes.append(self)