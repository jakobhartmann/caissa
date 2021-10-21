from datetime import date

import numpy as np
import chess
from chess import variant, pgn

from three_check_chess.config import Config


'''Wrapper class giving access to the game specific logic of Three-Check Chess through generic functions'''
class GameBoard:
    '''Initialize the class variables'''
    def __init__(self, board = None, move_threshold = None):
        # If a board is given, use it, else initialize a new three-check chess board
        self.board = board if board != None else variant.ThreeCheckBoard()
        # Move temperature to trade-off exploration and exploitation during move selection
        self.tau = Config.TAU
        self.move_threshold = move_threshold if move_threshold != None else np.random.normal(Config.MOVE_THRESHOLD_MU, Config.MOVE_THRESHOLD_SIGMA)
        

    '''Returns the string representation of the board position'''
    def __str__(self):
        # Return the string representation of the plain three-check chess board
        return self.board.__str__()


    '''Returns a new GameBoard instance with a deepcopy of the current three-check chess board'''
    def __deepcopy__(self, memo):
        return GameBoard(self.board.copy(), self.move_threshold)
            

    '''Make the given move on the board'''
    def make_move(self, move):
        self.board.push(move)


    '''Returns the side to move'''
    def get_turn(self):
        return self.board.turn


    '''Returns the encoded result of the game'''
    def get_result(self):
        # Get the outcome of the game
        result = self.board.outcome()
        # If the game is not over yet, return None
        if result == None:
            return None
        # Else get the winner of the game
        else:
            winner = result.winner
        
        # Return the encoded result of the game as 1 (white won), -1 (black won) or 0 (game ended in a draw)
        if winner == True:
            return 1
        elif winner == False:
            return -1
        elif winner == None:
            return 0


    '''Returns a list of all legal moves in the current position'''
    def get_legal_moves(self):
        return self.board.legal_moves


    '''Returns the last move of the game'''
    def get_last_move(self):
        # If the move stack is empty, return None, else return the last move
        return self.board.peek() if len(self.board.move_stack) > 0 else None
    

    '''Returns the history of the current game'''
    def get_game_history(self, event, round, player1, player2):
        # Create a new root node of a PGN game
        game = chess.pgn.Game()
        node = game
        # Add each move from the move stack to the PGN
        for move in self.board.move_stack:
            node = node.add_variation(move)
        # Add the given header information together with the date and result of the game to the PGN
        game.headers['Event'] = event
        game.headers['Date'] = date.today().strftime('%Y.%m.%d')
        game.headers['Round'] = round
        game.headers['White'] = player1
        game.headers['Black'] = player2
        game.headers['Result'] = self.board.result()
        # Get the string representation of the PGN
        game_history = str(game)
        # Add two blank lines to separate the games in the PGN file
        game_history += '\n\n'
        # Return the game history
        return game_history


    '''Returns the move temperature based on the number of moves played in the current game'''
    def get_move_temperature(self):
        # If the number of moves played in the current game is smaller or equal to the move threshold, return tau
        if self.board.ply() <= self.move_threshold:
            return self.tau
        # Else return None, which leads to applying the argmax operator and thus picking the most visited node
        else:
            return None