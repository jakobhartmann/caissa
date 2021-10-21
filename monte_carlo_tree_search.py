import concurrent
from concurrent.futures.thread import ThreadPoolExecutor
import threading
import copy
import traceback

import numpy as np
from sklearn import preprocessing

from node import Node
from neural_network import NeuralNetwork

# from tic_tac_toe.config import Config
# from tic_tac_toe.board_utils import BoardUtils

from three_check_chess.config import Config
from three_check_chess.board_utils import BoardUtils


'''Performs Monte Carlo Tree Search using a neural network for reducing the depth and breadth of the search tree'''
class MonteCarloTreeSearch:
    '''Initialize the class variables'''
    def __init__(self, transposition_table, model = None):
        # Transposition table to avoid querying the neural network multiple times for the same position
        self.transposition_table = transposition_table
        # Maximum number of entries in the transposition table
        self.max_num_entries_transposition_table = Config.MAX_NUM_ENTRIES_TRANSPOSITION_TABLE
        # If a model is given, it is used for inference, otherwise TensorFlow Serving is used for predictions
        self.model = model
        # Number of possible legal and illegal moves that the neural network can represent
        self.num_moves = Config.NUM_MOVES
        # Neural network class giving access to the prediction API
        self.neural_network = NeuralNetwork()
        # BoardUtils class which can generate bitboards and assign moves to their corresponding policy indices and vice versa
        self.board_utils = BoardUtils()
        # Exploration rates for the PUCT score
        self.c_base = Config.C_BASE
        self.c_init = Config.C_INIT
        # Dirichlet noise which is added to the root node
        self.alpha = Config.ALPHA
        self.epsilon = Config.EPSILON
        # Number of threads used for tree parallelization
        self.num_threads = Config.NUM_THREADS
        # Virtual loss which is added during tree traversal
        self.virtual_loss = Config.VIRTUAL_LOSS
        # Lock to prevent that multiple threads modify the search tree at the same time
        self.lock = threading.Lock()


    '''Returns the root node after the given number of iterations'''
    def search(self, board, num_iterations, root = None):
        # Initialize the stop variable, which can be changed from outside to stop the execution
        self.stop = False

        # If no root node is given, create a new one with the current board and the corresponding zobrist key
        if root == None:
            # If this is a new game, get the initial zobrist key, else calculate it
            zobrist_key = Config.INITIAL_ZOBRIST_KEY if board.get_last_move() == None else self.transposition_table.get_zobrist_key(board)
            self.root = Node(board = board, zobrist_key = zobrist_key, parent = None, move = None, prior_probability = None)
        # Else reuse part of the old search tree
        else:
            # Initialize the new root node with the given one
            self.root = root
            # Set the prior probabilities and the parent of the new root node to None
            self.root.parent = None
            self.prior_probability = None
            # Add dirichlet noise to the prior probabilities of the children of the root node
            dirichlet_noise = np.random.dirichlet([self.alpha for _ in range(len(self.root.children))])
            for i in range(len(self.root.children)):
                self.root.children[i].prior_probability = (1 - self.epsilon) * self.root.children[i].prior_probability + self.epsilon * dirichlet_noise[i]
            # Substract the number of visits of the root node from the number of iterations to reduce the search time and run MCTS for the given number of iterations
            num_iterations -= self.root.visits

        # Initialize a ThreadPoolExecutor for tree parallelization
        with ThreadPoolExecutor(self.num_threads) as executor:
            # Dict used to store possible exceptions occuring in the threads
            futures = {}
            # Run MCTS for the given number of iterations
            for iteration in range(num_iterations):
                # Save the result of the execution
                futures[iteration] = executor.submit(self.simulation)

            # source: https://stackoverflow.com/a/58766392
            # Iterate over all results and print possible exceptions
            for future in concurrent.futures.as_completed(futures[key] for key in futures.keys()):
                if future.exception():
                    print('Exception: ', future.exception())
                    
        # Return the root at the end of the search
        return self.root


    '''Performs one iteration of MCTS'''
    def simulation(self):
        try:
            # Do not execute the MCTS iteration if the stop variable has been changed to True
            if self.stop:
                return

            # Get the leaf node based on the PUCT score
            leaf = self.select(self.root)
            # While the leaf node is None (because it is already visited by another thread), repeat the selection phase
            while leaf == None:
                leaf = self.select(self.root)

            # If the selected leaf node has no board, make a deepcopy of the parent board and calculate the new position and zobrist key
            # This ensures that the expensive copy operation is only performed for nodes which are expanded during the MCTS iteration, which is a small subset of all nodes in the search tree and thus reduces the search time significantly
            if leaf.board == None:
                self.lock.acquire()
                leaf_board = copy.deepcopy(leaf.parent.board)
                leaf_board.make_move(leaf.move)
                leaf_zobrist_key = self.transposition_table.update_zobrist_key(zobrist_key = leaf.parent.zobrist_key, old_board = leaf.parent.board, new_board = leaf_board, move = leaf.move)
                leaf.board = leaf_board
                leaf.zobrist_key = leaf_zobrist_key
                self.lock.release()

            # If the game has not ended, expand the leaf node and get the value of the board position
            if leaf.board.get_result() == None:
                value = self.expand(leaf)
            # Else, if the game ended in a draw set the value to 0, otherwise the side which played the last move won
            else:
                value = 0 if leaf.board.get_result() == 0 else 1

            # Backpropagate the value from the leaf to the root and remove the virtual loss
            self.lock.acquire()
            self.backpropagate(node = leaf, visit = -self.virtual_loss + 1, value = value, virtual_loss_counter = -1)
            self.lock.release()
        except Exception:
            print(traceback.format_exc())


    '''Returns the leaf node based on the PUCT score'''
    def select(self, node):
        # Traverse the search tree until a leaf node is found
        while len(node.children) > 0:
            self.lock.acquire()
            # Add a virtual loss to the visits of the node to discourage other threads from taking the same path
            node.visits += self.virtual_loss
            # Increment the virtual loss counter by one to indicate that this node is currently part of a tree traversal by another thread
            node.virtual_loss_counter += 1
            self.lock.release()

            # Determine the PUCT score for each child node and pick the node with the highest score for the next iteration of the tree traversal
            children_with_puct_scores = {self.puct(child): child for child in node.children}
            node = children_with_puct_scores[max(children_with_puct_scores.keys())]

        # Also increase the visits and virtual loss counter of the selected leaf node
        self.lock.acquire()
        node.visits += self.virtual_loss
        node.virtual_loss_counter += 1
        self.lock.release()

        # If this leaf node is not currently visited by another thread and was not expanded in the meantime, return it
        if node.virtual_loss_counter == 1 and len(node.children) == 0:
            return node
        # Else, remove the virtual loss from all nodes encountered in the tree traversal and return None
        else:
            self.lock.acquire()
            self.backpropagate(node = node, visit = -self.virtual_loss, value = 0, virtual_loss_counter = -1)
            self.lock.release()
            return None


    '''Returns the PUCT score of the given node'''
    def puct(self, node):
        # Calculate the exploration rate
        c = np.log((1 + node.parent.visits + self.c_base) / self.c_base) + self.c_init
        # Calculate the exploration term
        U = c * node.prior_probability * (np.sqrt(node.parent.visits) / (1 + node.visits))
        # Return the exploitation term (mean value) plus the exploration term
        return node.mean_value + U


    '''Expands the given node with all legal moves and returns the value of its board position'''
    def expand(self, node):
        # Create a dictionary with all legal moves in the current position together with their corresponding policy indices 
        turn = node.board.get_turn()
        legal_moves_with_indices = {move: self.board_utils.get_policy_index_from_move(move, turn) for move in node.board.get_legal_moves()}

        # Get the prior probabilities and value of the current board position from the neural network
        policy, value = self.get_neural_network_evaluation(node.board, node.zobrist_key, legal_moves_with_indices.values())

        # For each legal move, create a new node and add it to the list of children
        # IMPORTANT!!! The new nodes only contain information about their corresponding move and prior probability, but not about the board position or the zobrist key!
        # These information are only added when a node is selected for the expansion-step, thus expensive operations are minimized and the search time is reduced significantly.
        for move, index in legal_moves_with_indices.items():
            new_node = Node(board = None, zobrist_key = None, parent = node, move = move, prior_probability = policy[index])
            node.children.append(new_node)

        # If the given node is the root, add dirichlet noise to the prior probabilities of the children
        if node.parent == None:
            dirichlet_noise = np.random.dirichlet([self.alpha for _ in range(len(node.children))])
            for i in range(len(node.children)):
                node.children[i].prior_probability = (1 - self.epsilon) * node.children[i].prior_probability + self.epsilon * dirichlet_noise[i]

        # Return the value of the board position of the given leaf node
        return value


    '''Backpropagates the value from the leaf to the root together with increasing/decreasing the visits/virtual loss counter'''
    def backpropagate(self, node, visit, value, virtual_loss_counter):
        # Traverse the search tree from bottom to top and update the node statistics
        while node != None:
            # Update the visit count
            node.visits += visit
            # Update the value of the node with the result of the game or the value obtained from the neural network
            node.value += value
            # Recalculate the mean value (number of visits can be 0, if this function is only used to remove the virtual loss!)
            node.mean_value = 0 if node.visits == 0 else node.value / node.visits
            # Update (decrease) the virtual loss counter
            node.virtual_loss_counter += virtual_loss_counter
            # Negate the value and move one node up in the search tree by selecting the parent of the current node to be the one updated in the next iteration
            value *= -1
            node = node.parent


    '''Returns the (normalized) prior probabilities of all legal moves and the value of the given board position'''
    def get_neural_network_evaluation(self, board, zobrist_key = None, indices = None):
        # If a zobrist key was given and the position was already encountered, return the result from the transposition table
        if zobrist_key != None and zobrist_key in self.transposition_table.hash_table:
            return self.transposition_table.hash_table[zobrist_key]

        # If the policy indices of the legal moves are not given, calculate them
        if indices == None:
            turn = board.get_turn()
            indices = [self.board_utils.get_policy_index_from_move(move, turn) for move in board.get_legal_moves()]

        # Get the bitboard of the current position
        bitboard = self.board_utils.get_bitboard(board)

        # If no model was given, use the gRPC-API from TensorFlow Serving for model inference
        if self.model == None:
            nn_policy, nn_value = self.neural_network.make_grpc_prediction(bitboard)
        # Else, get the policy and value directly from the given model
        else:
            nn_policy, nn_value = self.model.predict(bitboard)
            nn_value = nn_value[0][0]

        nn_policy = nn_policy[0]
        
        '''# Alternative where first the illegal moves are masked out and then the softmax activation function is applied (changes in the neural network class necessary!)
        # This causes the policy distribution to be more uneven (which is logical), but also causes the policy loss to drop to zero during training and the network returning bad results. --> The reason for this is not clear yet.
        # Initialize all values of the policy vector with negative infinity to ensure that all illegal moves have a probability of 0 after applying the softmax activation function
        policy = [-np.inf for _ in range(self.num_moves)]
        # For each legal move, set the corresponding value in the policy vector to the output of the neural network
        for index in indices:
            policy[index] = nn_policy[index]
        # Apply the softmax activation function to get the final probabilities. 
        # Increase the numerical stability and prevent the exponentiated values from exploding by subtracting the maximum from them first. This does not change the result (source: https://cs231n.github.io/linear-classify).
        policy -= np.max(policy)
        policy = np.exp(policy)
        policy /= np.sum(policy)'''

        # Initialize the policy vector
        policy = np.zeros(self.num_moves)
        # For each legal move, set the corresponding value in the policy vector to the output of the neural network
        for index in indices:
            policy[index] = nn_policy[index]

        # Use a uniform policy in case the neural network has returned a probability of 0 for all legal moves
        if sum(policy) == 0:
            for index in indices:
                policy[index] = 1

        # Normalize the policy vector to unit norm
        policy = preprocessing.normalize([policy], norm = 'l1')[0]

        # If the zobrist key is not None and the maximum number of entries in the transposition table is not reached, add the policy and value to the transposition table
        if zobrist_key != None and len(self.transposition_table.hash_table) < self.max_num_entries_transposition_table:
            self.transposition_table.hash_table[zobrist_key] = (policy, nn_value)

        # Return the policy and value
        return policy, nn_value


    '''Returns a probability distribution corresponding to the proportional number of visits for each move/child node of the root node'''
    def get_move_probabilities(self, root):
        # Initialize the probability for each move to 0
        probabilities = np.zeros(self.num_moves)

        # For each child of the root node calculate the move probabilities by dividing the number of visits of that child node by the total number of visits of all child nodes of the root
        turn = root.board.get_turn()
        for child in root.children:
            index = self.board_utils.get_policy_index_from_move(child.move, turn)
            probabilities[index] = child.visits / (root.visits - 1)

        # Return the probability distribution
        return probabilities

    
    '''Returns the next move either deterministically or stochastically together with the corresponding child node'''
    def move_selection(self, root, exploratory):
        # If the move should be determined stochastically
        if exploratory:
            # Initialize the move probability for each move to 0
            move_probabilities = np.zeros(self.num_moves)
            # Get the move temperature
            tau = root.board.get_move_temperature()
            # If tau == None, return the most visited node
            if tau == None:
                return self.move_selection(root, False)
            # Exponentiate the number of visits of each child by 1 / tau
            turn = root.board.get_turn()
            for child in root.children:
                index = self.board_utils.get_policy_index_from_move(child.move, turn)
                move_probabilities[index] = child.visits ** (1 / tau)
            # Normalize the move probabilities to unit norm
            move_probabilities = preprocessing.normalize([move_probabilities], norm = 'l1')[0]
            # Stochastically pick a move index based on the calculated probability distribution
            index = np.random.choice(np.arange(self.num_moves), p = move_probabilities)
        # Else get the move probabilities and determinstically pick the move index with the most visits
        else:
            index = np.argmax(self.get_move_probabilities(root))
        
        # Get the move corresponding to the obtained index
        move = self.board_utils.get_move_from_policy_index(index, root.board.get_turn())

        # Find the child node which corresponds to the chosen move and return it together with the move
        for child in root.children:
            if child.move == move:
                return (child, move)