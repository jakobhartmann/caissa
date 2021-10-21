import numpy as np


'''Represents a node in the MCTS search tree'''
class Node:
    '''Initialize the class variables'''
    def __init__(self, board, zobrist_key, parent, move, prior_probability):
        # Parent of the node
        self.parent = parent
        # List of child nodes
        self.children = []
        # Value
        self.value = 0
        # Number of visits
        self.visits = 0
        # Mean value = value / visits
        self.mean_value = 0
        # Prior probability obtained from the neural network
        self.prior_probability = prior_probability
        # Virtual loss counter to enable MCTS tree paralellization
        self.virtual_loss_counter = 0
        # Move associated with this node
        self.move = move
        # Board associated with this node
        self.board = board
        # Zobrist key of the board
        self.zobrist_key = zobrist_key


    '''String representation of a node'''
    def __str__(self):
        result = ''
        result += 'Parent: ' + repr(self.parent) + '\n'
        result += 'Children: ' + repr(self.children) + '\n'
        result += 'Value: ' + str(self.value) + '\n'
        result += 'Visits: ' + str(self.visits) + '\n'
        result += 'Mean Value: ' + str(self.mean_value) + '\n'
        result += 'Prior Probability: ' + str(self.prior_probability) + '\n'
        result += 'Virtual Loss Counter: ' + str(self.virtual_loss_counter) + '\n'
        result += 'Move: ' + str(self.move) + '\n'
        result += 'Board: \n' + str(self.board) + '\n'
        result += 'Zobrist Key: ' + str(self.zobrist_key) + '\n'
        return result


# source: https://stackoverflow.com/questions/20242479/printing-a-tree-data-structure-in-python
'''Print the search tree starting with the given node and ending at the given depth'''
def print_tree(node, string = '', depth = 0, max_depth = np.inf):
    if depth >= max_depth:
        return

    print('\t' * depth + 'Value: ' + str(node.value) + ', Visits: ' + str(node.visits) + ', Mean Value: ' + str(node.mean_value) + ', Prior Probability: ' + str(node.prior_probability) + ', Virtual Loss Counter: ' + str(node.virtual_loss_counter) + ', Move: ' + str(node.move) + ', Zobrist Key: ' + str(node.zobrist_key))

    for child in node.children:
        print_tree(child, string, depth + 1, max_depth)


'''Returns the number of nodes in the (sub-)tree starting from the given node'''
def get_tree_size(node):
    if node.children == []:
        return 1
    else:
        return 1 + sum(get_tree_size(child) for child in node.children)