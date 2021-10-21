import numpy as np

import chess

from three_check_chess.config import Config


'''Stores the transposition table and its logic for Three-Check Chess'''
class TranspositionTable:
    '''Initialize the class variables'''
    def __init__(self):
        # Seed for the random generator
        self.random_generator_seed = Config.RANDOM_GENERATOR_SEED
        # Initialize the random generator to ensure the same zobrist values accross different machines
        self.rng = np.random.default_rng(self.random_generator_seed)
        # Generate a random number for each piece at each possible square plus numbers for en passant, castling rights, remaining checks, the side to move and the repetition count
        self.zobrist_values = self.rng.integers(0, 2**63, size = (66, 12), dtype = np.int64)
        # Assign each piece a unique number to access the zobrist values
        self.piece_indices = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
        # Whether or not to include the repetition count in the zobrist key
        self.include_repetition_count = Config.INCLUDE_REPETITION_COUNT
        # Initialize the hash table storing the zobrist/hash keys together with the corresponding neural network evaluations
        self.hash_table = {}


    '''Returns the zobrist/hash key for the given board position'''
    def get_zobrist_key(self, board):
        # Get the plain three-check chess board
        board = board.board

        # Initialize the zobrist key
        zobrist_key = 0

        # Iterate over all squares and XOR the zobrist values for each non-empty square and its respective piece with the zobrist key
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece != None:
                zobrist_key ^= self.zobrist_values[square][self.piece_indices[piece.symbol()]]

        # If an en passant capture is possible XOR the zobrist values of the corresponding file to the zobrist key
        if board.has_legal_en_passant():
            zobrist_key ^= self.zobrist_values[64][chess.square_file(board.ep_square)]

        # Determine for each side whether kingside and queenside castling is still possible and add the information to the zobrist key
        if board.has_kingside_castling_rights(chess.WHITE):
            zobrist_key ^= self.zobrist_values[64][8]
        if board.has_queenside_castling_rights(chess.WHITE):
            zobrist_key ^= self.zobrist_values[64][9]
        if board.has_kingside_castling_rights(chess.BLACK):
            zobrist_key ^= self.zobrist_values[64][10]
        if board.has_queenside_castling_rights(chess.BLACK):
            zobrist_key ^= self.zobrist_values[64][11]

        # Determine for each side how many more checks are necessary to win the game and add the information to the zobrist key
        zobrist_key ^= self.zobrist_values[65][board.remaining_checks[chess.WHITE]]
        zobrist_key ^= self.zobrist_values[65][board.remaining_checks[chess.BLACK] + 4]

        # Add the information to the zobrist key whether it is black to move
        if not board.turn:
            zobrist_key ^= self.zobrist_values[65][8]

        # If specified, add the information about the repetition count
        if self.include_repetition_count:
            if board.is_repetition(2):
                zobrist_key ^= self.zobrist_values[65][9]
                if board.is_repetition(3):
                    zobrist_key ^= self.zobrist_values[65][9]
                    zobrist_key ^= self.zobrist_values[65][10]
                    if board.is_repetition(4):
                        zobrist_key ^= self.zobrist_values[65][10]
                        zobrist_key ^= self.zobrist_values[65][11]

        # Return the zobrist key
        return zobrist_key


    '''Returns the zobrist/hash key for a new position given the old (copied) board and the played move without iterating over the whole board'''
    def update_zobrist_key(self, zobrist_key, old_board, new_board, move):    
        # Get the plain three-check chess boards
        old_board = old_board.board
        new_board = new_board.board

        ### Update pieces ###
        # Get the pieces on the starting and target square before and after the move
        piece_at_starting_square = old_board.piece_at(move.from_square)
        new_piece_at_target_square = new_board.piece_at(move.to_square)
        old_piece_at_target_square = old_board.piece_at(move.to_square)

        # If a piece was captured, remove it from the target square
        if old_piece_at_target_square != None:
            zobrist_key ^= self.zobrist_values[move.to_square][self.piece_indices[old_piece_at_target_square.symbol()]]
        # Elif the move was an en passant capture, remove the respective pawn from the board
        elif old_board.is_en_passant(move):
            ep_square = old_board.ep_square + 8 if new_board.turn else old_board.ep_square - 8
            zobrist_key ^= self.zobrist_values[ep_square][self.piece_indices[chess.Piece(chess.PAWN, new_board.turn).symbol()]]
        # Elif the move was a castling move, also move the rook from the starting square to the target square
        elif old_board.is_castling(move):
            if old_board.is_kingside_castling(move):
                starting_square = chess.H1 if old_board.turn else chess.H8
                target_square = chess.F1 if old_board.turn else chess.F8
            else:
                starting_square = chess.A1 if old_board.turn else chess.A8
                target_square = chess.D1 if old_board.turn else chess.D8
            zobrist_key ^= self.zobrist_values[starting_square][self.piece_indices[chess.Piece(chess.ROOK, old_board.turn).symbol()]]
            zobrist_key ^= self.zobrist_values[target_square][self.piece_indices[chess.Piece(chess.ROOK, old_board.turn).symbol()]]

        # Remove the piece from the starting square and add it (or the promoted piece) to the target square
        zobrist_key ^= self.zobrist_values[move.from_square][self.piece_indices[piece_at_starting_square.symbol()]]
        zobrist_key ^= self.zobrist_values[move.to_square][self.piece_indices[new_piece_at_target_square.symbol()]]

        ### Update en passant ###
        # If there was a legal en passant capture in the last position, remove this information from the zobrist key
        if old_board.has_legal_en_passant():
            zobrist_key ^= self.zobrist_values[64][chess.square_file(old_board.ep_square)]
        
        # If an en passant capture is possible, XOR the zobrist values of the corresponding file to the zobrist key
        if new_board.has_legal_en_passant():
            zobrist_key ^= self.zobrist_values[64][chess.square_file(new_board.ep_square)]

        ### Update castling rights ###
        # If the castling rights have changed with the last move, remove the respective information from the zobrist key
        if old_board.has_kingside_castling_rights(chess.WHITE) and not new_board.has_kingside_castling_rights(chess.WHITE):
            zobrist_key ^= self.zobrist_values[64][8]
        if old_board.has_queenside_castling_rights(chess.WHITE) and not new_board.has_queenside_castling_rights(chess.WHITE):
            zobrist_key ^= self.zobrist_values[64][9]
        if old_board.has_kingside_castling_rights(chess.BLACK) and not new_board.has_kingside_castling_rights(chess.BLACK):
            zobrist_key ^= self.zobrist_values[64][10]
        if old_board.has_queenside_castling_rights(chess.BLACK) and not new_board.has_queenside_castling_rights(chess.BLACK):
            zobrist_key ^= self.zobrist_values[64][11]

        ### Update remaining checks ###
        # If the last move was a check, update the information about the remaining checks
        if new_board.is_check():
            color_shift = 0 if old_board.turn else 4
            zobrist_key ^= self.zobrist_values[65][old_board.remaining_checks[old_board.turn] + color_shift]
            zobrist_key ^= self.zobrist_values[65][new_board.remaining_checks[old_board.turn] + color_shift]

        ### Update the side to move ###
        zobrist_key ^= self.zobrist_values[65][8]

        ### Update repetition count ###
        if self.include_repetition_count:
            # Remove repetition count from the last position
            if old_board.is_repetition(2):
                zobrist_key ^= self.zobrist_values[65][9]
                if old_board.is_repetition(3):
                    zobrist_key ^= self.zobrist_values[65][9]
                    zobrist_key ^= self.zobrist_values[65][10]
                    if old_board.is_repetition(4):
                        zobrist_key ^= self.zobrist_values[65][10]
                        zobrist_key ^= self.zobrist_values[65][11]

            # Add repetition count for the new position
            if new_board.is_repetition(2):
                zobrist_key ^= self.zobrist_values[65][9]
                if new_board.is_repetition(3):
                    zobrist_key ^= self.zobrist_values[65][9]
                    zobrist_key ^= self.zobrist_values[65][10]
                    if new_board.is_repetition(4):
                        zobrist_key ^= self.zobrist_values[65][10]
                        zobrist_key ^= self.zobrist_values[65][11]

        # Return the updated zobrist key
        return zobrist_key