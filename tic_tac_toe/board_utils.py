import numpy as np

from tic_tac_toe.config import Config


'''Utility class which provides helper functions for generating the input representation of the neural network and transforming its action output'''
class BoardUtils:
    '''Initialize the class variables'''
    def __init__(self):
        # Number of possible legal and illegal moves that the neural network can represent
        self.num_moves = Config.NUM_MOVES


    '''Returns the bitboard representation of the given position'''
    def get_bitboard(self, board):
        # Initialize the plane representing the player whose turn it is
        p1 = np.zeros(self.num_moves)
        # Initialize the plane representing the player whose turn it is not
        p2 = np.zeros(self.num_moves)

        # Iterate over all squares and if a square is occupied by a player, set the entry in the corresponding plane to 1
        for i in range(self.num_moves):
            if board.board[i] == board.turn:
                p1[i] = 1
            elif board.board[i] == board.not_turn:
                p2[i] = 1

        # Plane representing the side to make the next move (all ones for x or all zeros for o)
        if board.turn == 'x':
            p3 = np.ones(self.num_moves)
        else:
            p3 = np.zeros(self.num_moves)

        # Reshape all planes to 3x3
        p1 = np.reshape(p1, (3, 3))
        p2 = np.reshape(p2, (3, 3))
        p3 = np.reshape(p3, (3, 3))

        # Stack all planes together along axis 2 to get the data in NHWC format (expected by the neural network!)
        bitboard = np.stack([p1, p2, p3], axis = 2)

        # Return the bitboard
        return np.array([bitboard])


    '''Returns the policy index corresponding to the given move'''
    def get_policy_index_from_move(self, move, turn):
        return move


    '''Returns the move corresponding to the given policy index'''
    def get_move_from_policy_index(self, index, turn):
        return index