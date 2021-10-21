import numpy as np

import chess

from three_check_chess.config import Config


'''Utility class which provides helper functions for mirroring moves, generating the input representation of the neural network and transforming its action output'''
class BoardUtils:
    '''Mirrors the given move vertically'''
    def mirror_move(self, move):
        return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)

    '''Returns the bitboard representation for the given chess board'''
    def get_bitboard(self, board):
        # Get the plain three-check chess board
        board = board.board
        
        # Plane with all zeros if it is white's turn and all ones if it is black's turn 
        side_to_move = np.full((8, 8), int(not board.turn))

        # Since checking for a threefold repetition in the outcome function of python-chess is expensive and offering draws is not possible, only fivefold repetitions are considered an automatic draw
        # Since the neural network is only given non-terminal chess positions, a fivefold repetition plane is not included
        second_occurrence = np.zeros((8, 8))
        third_occurrence = np.zeros((8, 8))
        fourth_occurrence = np.zeros((8, 8))

        # Each plane indicates whether the current position is occurring exactly for the first, second, third or fourth time respectively
        first_occurrence = np.ones((8, 8))
        if board.is_repetition(2):
            first_occurrence = np.zeros((8, 8))
            second_occurrence = np.ones((8, 8))
            if board.is_repetition(3):
                second_occurrence = np.zeros((8, 8))
                third_occurrence = np.ones((8, 8))
                if board.is_repetition(4):
                    third_occurrence = np.zeros((8, 8))
                    fourth_occurrence = np.ones((8, 8))

        # Plane with the number of plies / half-moves (the number of plies is used instead of the full move count to be consistent with the no progress count)
        plies = np.full((8, 8), board.ply()) 

        # Plane with the number of half-moves since the last capture or pawn move (relevant for the seventyfive-move rule, i.e. 150 half-moves)
        no_progress_count = np.full((8, 8), board.halfmove_clock)

        # Each of the three planes indicates whether the side to move has to give exactly three, two or one more check(s) to win the game
        p1_three_checks_remaining = np.full((8, 8), int(board.remaining_checks[board.turn] == 3))
        p1_two_checks_remaining = np.full((8, 8), int(board.remaining_checks[board.turn] == 2))
        p1_one_check_remaining = np.full((8, 8), int(board.remaining_checks[board.turn] == 1))

        # Eaach of the three planes indicates whether the side not to move has to give exactly three, two or one more check(s) to win the game
        p2_three_checks_remaining = np.full((8, 8), int(board.remaining_checks[not board.turn] == 3))
        p2_two_checks_remaining = np.full((8, 8), int(board.remaining_checks[not board.turn] == 2))
        p2_one_check_remaining = np.full((8, 8), int(board.remaining_checks[not board.turn] == 1))

        # Mirror the board vertically if it is black's turn to always show the board from the point of view of the player whose turn it is
        # IMPORTANT!!! This can cause problems with other game statistics like repetition count or number of remaining checks!!!
        if not board.turn:
            board_from_white_side = board.mirror()
        else:
            board_from_white_side = board

        # Create one bitboard for each piece type of white
        p1_pawns = np.array(board_from_white_side.pieces(chess.PAWN, chess.WHITE).tolist(), int).reshape((8, 8))
        p1_knights = np.array(board_from_white_side.pieces(chess.KNIGHT, chess.WHITE).tolist(), int).reshape((8, 8))
        p1_bishops = np.array(board_from_white_side.pieces(chess.BISHOP, chess.WHITE).tolist(), int).reshape((8, 8))
        p1_rooks = np.array(board_from_white_side.pieces(chess.ROOK, chess.WHITE).tolist(), int).reshape((8, 8))
        p1_queens = np.array(board_from_white_side.pieces(chess.QUEEN, chess.WHITE).tolist(), int).reshape((8, 8))
        p1_king = np.array(board_from_white_side.pieces(chess.KING, chess.WHITE).tolist(), int).reshape((8, 8))

        # Create one bitboard for each piece type of black
        p2_pawns = np.array(board_from_white_side.pieces(chess.PAWN, chess.BLACK).tolist(), int).reshape((8, 8))
        p2_knights = np.array(board_from_white_side.pieces(chess.KNIGHT, chess.BLACK).tolist(), int).reshape((8, 8))
        p2_bishops = np.array(board_from_white_side.pieces(chess.BISHOP, chess.BLACK).tolist(), int).reshape((8, 8))
        p2_rooks = np.array(board_from_white_side.pieces(chess.ROOK, chess.BLACK).tolist(), int).reshape((8, 8))
        p2_queens = np.array(board_from_white_side.pieces(chess.QUEEN, chess.BLACK).tolist(), int).reshape((8, 8))
        p2_king = np.array(board_from_white_side.pieces(chess.KING, chess.BLACK).tolist(), int).reshape((8, 8))

        # Create two bitboards for the kingside and queenside castling rights of white
        p1_kingside_castling_rights = np.full((8, 8), int(board_from_white_side.has_kingside_castling_rights(chess.WHITE)))
        p1_queenside_castling_rights = np.full((8, 8), int(board_from_white_side.has_queenside_castling_rights(chess.WHITE)))
        
        # Create two bitboards for the kingside and queenside castling rights of black
        p2_kingside_castling_rights = np.full((8, 8), int(board_from_white_side.has_kingside_castling_rights(chess.BLACK)))
        p2_queenside_castling_rights = np.full((8, 8), int(board_from_white_side.has_queenside_castling_rights(chess.BLACK)))

        # If the side to move has a legal en passant move, the en passant / target square is marked with a one
        if board_from_white_side.has_legal_en_passant():
            en_passant = np.zeros(64)
            en_passant[board_from_white_side.ep_square] = 1
            en_passant = np.reshape(en_passant, (8, 8))
        else:
            en_passant = np.zeros((8, 8))

        # Stack all planes together along axis 2 to get the data in NHWC format (expected by the neural network!)
        # The alternative of stacking all planes together along axis 0 and then transforming them with tf.transpose(bitboard, [0, 2, 3, 1]) to NHWC format is significantly slower!!!
        bitboard = np.stack([p1_pawns, p1_knights, p1_bishops, p1_rooks, p1_queens, p1_king, 
                            p2_pawns, p2_knights, p2_bishops, p2_rooks, p2_queens, p2_king, 
                            p1_kingside_castling_rights, p1_queenside_castling_rights, 
                            p2_kingside_castling_rights, p2_queenside_castling_rights, 
                            en_passant, first_occurrence, second_occurrence, third_occurrence, fourth_occurrence, side_to_move, 
                            p1_three_checks_remaining, p1_two_checks_remaining, p1_one_check_remaining, 
                            p2_three_checks_remaining, p2_two_checks_remaining, p2_one_check_remaining, 
                            plies, no_progress_count], axis = 2)

        # Return the bitboard
        return np.array([bitboard])


    '''Returns a dictionary with all possible legal and illegal UCI moves based on the output representation of the neural network'''
    def create_uci_moves(self):
        # Counter to assign each move a unique number to later access the flatten array
        move_number = 0

        # Dictionary saving the UCI representation of a move together with a unique number
        moves = {}

        ### Queen moves ###
        # Possible directions a queen (and partially a rook, bishop and pawn) can move in
        queen_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        # Distance between starting and target square
        distances = [i for i in range(1, 8)]

        # Iterate over all queen directions, distances, files and ranks to generate the respective moves
        for queen_direction in queen_directions:
            for distance in distances:
                for file in chess.FILE_NAMES:
                    for rank in chess.RANK_NAMES:

                        # The starting square from where to pick up the piece
                        starting_square = file + rank

                        # Depending on the direction and distance calculate the target square
                        if queen_direction == 'N':
                            target_square = file + str(int(rank) + distance)
                        elif queen_direction == 'NE':
                            target_square = chr(ord(file) + distance) + str(int(rank) + distance)
                        elif queen_direction == 'E':
                            target_square = chr(ord(file) + distance) + rank
                        elif queen_direction == 'SE':
                            target_square = chr(ord(file) + distance) + str(int(rank) - distance)
                        elif queen_direction == 'S':
                            target_square = file + str(int(rank) - distance)
                        elif queen_direction == 'SW':
                            target_square = chr(ord(file) - distance) + str(int(rank) - distance)
                        elif queen_direction == 'W':
                            target_square = chr(ord(file) - distance) + rank
                        elif queen_direction == 'NW':
                            target_square = chr(ord(file) - distance) + str(int(rank) + distance)

                        # Concatenate the starting and target square to get the UCI move
                        move = starting_square + target_square

                        # Add the move to the dictionary together with the corresponding move number
                        moves[move] = move_number

                        # Increment the move counter
                        move_number += 1

        ### Knight moves ###
        # Possible moves of a knight
        knight_directions = ['N2E1', 'N1E2', 'S1E2', 'S2E1', 'S2W1', 'S1W2', 'N1W2', 'N2W1']

        # Iterate over all knight directions, files and ranks to generate the respective moves
        for knight_direction in knight_directions:
            for file in chess.FILE_NAMES:
                for rank in chess.RANK_NAMES:

                    # The starting square from where to pick up the piece
                    starting_square = file + rank

                    # Depending on the knight move calculate the target square
                    if knight_direction == 'N2E1':
                        target_square = chr(ord(file) + 1) + str(int(rank) + 2)
                    elif knight_direction == 'N1E2':
                        target_square = chr(ord(file) + 2) + str(int(rank) + 1)
                    elif knight_direction == 'S1E2':
                        target_square = chr(ord(file) + 2) + str(int(rank) - 1)
                    elif knight_direction == 'S2E1':
                        target_square = chr(ord(file) + 1) + str(int(rank) - 2)
                    elif knight_direction == 'S2W1':
                        target_square = chr(ord(file) - 1) + str(int(rank) - 2)
                    elif knight_direction == 'S1W2':
                        target_square = chr(ord(file) - 2) + str(int(rank) - 1)
                    elif knight_direction == 'N1W2':
                        target_square = chr(ord(file) - 2) + str(int(rank) + 1)
                    elif knight_direction == 'N2W1':
                        target_square = chr(ord(file) - 1) + str(int(rank) + 2)

                    # Concatenate the starting and target square to get the UCI move
                    move = starting_square + target_square

                    # Add the move to the dictionary together with the corresponding move number
                    moves[move] = move_number

                    # Increment the move counter
                    move_number += 1

        ### Pawn promotions ###
        # Possible directions a pawn can move in
        pawn_directions = ['W', 'N', 'E']

        # Possible pieces a pawn (which reaches the opponent's back-rank) can be promoted to
        promotions = ['q', 'r', 'n', 'b']

        # Iterate over all pawn directions, pieces, files and ranks to generate the respective moves
        for pawn_direction in pawn_directions:
            for promotion in promotions:
                for file in chess.FILE_NAMES:
                    for rank in chess.RANK_NAMES:

                        # The starting square from where to pick up the piece
                        starting_square = file + rank

                        # Depending on the direction calculate the target square together with a piece to which the pawn will be promoted
                        if pawn_direction == 'W':
                            target_square = chr(ord(file) - 1) + str(int(rank) + 1) + promotion
                        elif pawn_direction == 'N':
                            target_square = file + str(int(rank) + 1) + promotion
                        elif pawn_direction == 'E':
                            target_square = chr(ord(file) + 1) + str(int(rank) + 1) + promotion

                        # Concatenate the starting and target square to get the UCI move
                        move = starting_square + target_square

                        # Add the move to the dictionary together with the corresponding move number
                        moves[move] = move_number

                        # Increment the move counter
                        move_number += 1

        # Return the dictionary with all the moves and their corresponding number
        return moves


    '''Returns a dictionary with all possible mirrored legal and illegal UCI moves based on the output representation of the neural network'''
    def create_mirrored_uci_moves(self):
        # Counter to assign each move a unique number to later access the flatten array
        move_number = 0

        # Dictionary saving the UCI representation of a move together with a unique number
        moves = {}

        ### Queen moves ###
        # Possible directions a queen (and partially a rook, bishop and pawn) can move in
        queen_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            
        # Distance between starting and target square
        distances = [i for i in range(1, 8)]

        # Iterate over all queen directions, distances, files and ranks to generate the respective moves
        for queen_direction in queen_directions:
            for distance in distances:
                for file in chess.FILE_NAMES:
                    for rank in chess.RANK_NAMES:

                        # Mirror the rank
                        rank = str(9 - int(rank))

                        # The starting square from where to pick up the piece
                        starting_square = file + rank

                        # Depending on the direction and distance calculate the target square
                        if queen_direction == 'N':
                            target_square = file + str(int(rank) - distance)
                        elif queen_direction == 'NE':
                            target_square = chr(ord(file) + distance) + str(int(rank) - distance)
                        elif queen_direction == 'E':
                            target_square = chr(ord(file) + distance) + rank
                        elif queen_direction == 'SE':
                            target_square = chr(ord(file) + distance) + str(int(rank) + distance)
                        elif queen_direction == 'S':
                            target_square = file + str(int(rank) + distance)
                        elif queen_direction == 'SW':
                            target_square = chr(ord(file) - distance) + str(int(rank) + distance)
                        elif queen_direction == 'W':
                            target_square = chr(ord(file) - distance) + rank
                        elif queen_direction == 'NW':
                            target_square = chr(ord(file) - distance) + str(int(rank) - distance)

                        # Concatenate the starting and target square to get the UCI move
                        move = starting_square + target_square

                        # Add the move to the dictionary together with the corresponding move number
                        moves[move] = move_number

                        # Increment the move counter
                        move_number += 1

        ### Knight moves ###
        # Possible moves of a knight
        knight_directions = ['N2E1', 'N1E2', 'S1E2', 'S2E1', 'S2W1', 'S1W2', 'N1W2', 'N2W1']

        # Iterate over all knight directions, files and ranks to generate the respective moves
        for knight_direction in knight_directions:
            for file in chess.FILE_NAMES:
                for rank in chess.RANK_NAMES:

                    # Mirror the rank
                    rank = str(9 - int(rank))

                    # The starting square from where to pick up the piece
                    starting_square = file + rank

                    # Depending on the knight move calculate the target square
                    if knight_direction == 'N2E1':
                        target_square = chr(ord(file) + 1) + str(int(rank) - 2)
                    elif knight_direction == 'N1E2':
                        target_square = chr(ord(file) + 2) + str(int(rank) - 1)
                    elif knight_direction == 'S1E2':
                        target_square = chr(ord(file) + 2) + str(int(rank) + 1)
                    elif knight_direction == 'S2E1':
                        target_square = chr(ord(file) + 1) + str(int(rank) + 2)
                    elif knight_direction == 'S2W1':
                        target_square = chr(ord(file) - 1) + str(int(rank) + 2)
                    elif knight_direction == 'S1W2':
                        target_square = chr(ord(file) - 2) + str(int(rank) + 1)
                    elif knight_direction == 'N1W2':
                        target_square = chr(ord(file) - 2) + str(int(rank) - 1)
                    elif knight_direction == 'N2W1':
                        target_square = chr(ord(file) - 1) + str(int(rank) - 2)

                    # Concatenate the starting and target square to get the UCI move
                    move = starting_square + target_square

                    # Add the move to the dictionary together with the corresponding move number
                    moves[move] = move_number

                    # Increment the move counter
                    move_number += 1

        ### Pawn promotions ###
        # Possible directions a pawn can move in
        pawn_directions = ['W', 'N', 'E']

        # Possible pieces a pawn (which reaches the opponent's back-rank) can be promoted to
        promotions = ['q', 'r', 'n', 'b']

        # Iterate over all pawn directions, pieces, files and ranks to generate the respective moves
        for pawn_direction in pawn_directions:
            for promotion in promotions:
                for file in chess.FILE_NAMES:
                    for rank in chess.RANK_NAMES:

                        # Mirror the rank
                        rank = str(9 - int(rank))

                        # The starting square from where to pick up the piece
                        starting_square = file + rank

                        # Depending on the direction calculate the target square together with a piece to which the pawn will be promoted
                        if pawn_direction == 'W':
                            target_square = chr(ord(file) - 1) + str(int(rank) - 1) + promotion
                        elif pawn_direction == 'N':
                            target_square = file + str(int(rank) - 1) + promotion
                        elif pawn_direction == 'E':
                            target_square = chr(ord(file) + 1) + str(int(rank) - 1) + promotion

                        # Concatenate the starting and target square to get the UCI move
                        move = starting_square + target_square

                        # Add the move to the dictionary together with the corresponding move number
                        moves[move] = move_number

                        # Increment the move counter
                        move_number += 1

        # Return the dictionary with all the moves and their corresponding number
        return moves

    
    '''Returns the policy index corresponding to the given move and side to play'''
    def get_policy_index_from_move(self, move, turn):
        # Depending on whose move it is, return the policy index from the corresponding dictionary
        # Alternative: if it´s blacks turn, mirror the move vertically and then look it up in the uci_moves_to_indices dictionary
        
        return Config.UCI_MOVES_TO_INDICES[move.uci()] if turn else Config.MIRRORED_UCI_MOVES_TO_INDICES[move.uci()]


    '''Returns the move corresponding to the given policy index and side to play'''
    def get_move_from_policy_index(self, index, turn):
        # Depending on whose move it is, get the move from the corresponding dictionary
        # Alternative: look up the move in the uci_moves_to_indices dictionary and if it´s blacks turn, mirror the move vertically
        
        uci_move = Config.INDICES_TO_UCI_MOVES[index] if turn else Config.INDICES_TO_MIRRORED_UCI_MOVES[index]
        
        # Parse the move to UCI notation
        return chess.Move.from_uci(uci_move)