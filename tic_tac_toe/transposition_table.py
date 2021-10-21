import numpy as np

from tic_tac_toe.config import Config


'''Stores the transposition table and its logic for Tic-Tac-Toe'''
class TranspositionTable:
    '''Initialize the class variables'''
    def __init__(self):
        # Number of possible legal and illegal moves that the neural network can represent
        self.num_moves = Config.NUM_MOVES
        # Seed for the random generator
        self.random_generator_seed = Config.RANDOM_GENERATOR_SEED
        # Initialize the random generator to ensure the same zobrist values accross different machines
        self.rng = np.random.default_rng(self.random_generator_seed)
        # Generate a random number for each piece at each possible square plus a number for the side to move
        self.zobrist_values = self.rng.integers(0, 2**31, size = (10, 2), dtype = np.int32)
        # Initialize the hash table storing the zobrist/hash keys together with the corresponding neural network evaluations
        self.hash_table = {}


    '''Returns the zobrist/hash key for the given board position'''
    def get_zobrist_key(self, board):
        # Initialize the zobrist key
        zobrist_key = 0

        # Iterate over all squares and XOR the zobrist values for each non-empty square and its respective nought or cross with the zobrist key
        for i in range(self.num_moves):
            if board.board[i] == 'x':
                zobrist_key ^= self.zobrist_values[i][0]
            elif board.board[i] == 'o':
                zobrist_key ^= self.zobrist_values[i][1]
            
        # Add the information to the zobrist key whether the side playing noughts is to move
        if board.turn == 'o':
            zobrist_key ^= self.zobrist_values[-1][1]
            
        # Return the zobrist key
        return zobrist_key


    '''Returns the zobrist key for a new position given the old zobrist key and the played move without iterating over the whole board'''
    def update_zobrist_key(self, zobrist_key, old_board, new_board, move):
        # Update the zobrist key with the new move
        # Important! The side to move already changed at this point!
        if new_board.turn == 'o':
            zobrist_key ^= self.zobrist_values[move][0]
        else:
            zobrist_key ^= self.zobrist_values[move][1]

        # Update the zobrist key with the player whose turn it is now
        zobrist_key ^= self.zobrist_values[-1][1]

        # Return the updated zobrist key
        return zobrist_key