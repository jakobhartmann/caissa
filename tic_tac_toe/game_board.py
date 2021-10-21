import numpy as np

from tic_tac_toe.config import Config


'''Wrapper class giving access to the game specific logic of Tic-Tac-Toe through generic functions'''
class GameBoard:
    '''Initialize the class variables'''
    def __init__(self):
        # Number of possible legal and illegal moves that the neural network can represent
        self.num_moves = Config.NUM_MOVES
        # Initialize an empty 3x3 game grid which is represented by a list of length 9
        self.board = ['.' for _ in range(self.num_moves)]
        # Initialize the legal moves at the beginning of a game
        self.legal_moves = [i for i in range(self.num_moves)]
        # Initialize the history of the current game
        self.game_history = []
        # List with both players
        self.players = ['x', 'o']
        # Player whose turn it is at the beginning of the game
        self.turn = 'x'
        # Player whose turn it is not at the beginning of the game
        self.not_turn = 'o'
        # Result of the game
        self.result = None
        # Move temperature to trade-off exploration and exploitation during move selection
        self.tau = Config.TAU
        self.move_threshold = np.random.normal(Config.MOVE_THRESHOLD_MU, Config.MOVE_THRESHOLD_SIGMA)


    '''Returns the string representation of the board position'''
    def __str__(self):
        result = ''
        # Print the 3x3 game grid with the moves played
        for i in range(self.num_moves):
            result += self.board[i]
            if i in [2, 5, 8]:
                result += '\n'
        return result
            

    '''Make the given move on the board and update related variables'''
    def make_move(self, move):
        # If the move is legal and the game is not over, make the move on the board
        if move in self.legal_moves and self.result == None:
            # Update the square on the board
            self.board[move] = self.turn
            # Remove the move from the list of legal ones and add it to the game history
            self.legal_moves.remove(move)
            self.game_history.append(move)
            # Flip the turn variables
            self.turn = 'x' if self.turn == 'o' else 'o'
            self.not_turn = 'x' if self.not_turn == 'o' else 'o'
            # Update the result
            self.result = self.update_result()
        else:
            print('ERROR! Move is either illegal or the game is already over!')


    '''Returns the side to move'''
    def get_turn(self):
        return self.turn


    '''Determines if the game is over and if so, which side won and returns the encoded result'''
    def update_result(self):
        # Initialize the result to None
        result = None
        # Check if a player has won the game, i.e., has occupied three squares either horizontally, vertically or diagonally
        for player in self.players:
            if ((self.board[0] == self.board[1] == self.board[2] == player) or 
                (self.board[3] == self.board[4] == self.board[5] == player) or 
                (self.board[6] == self.board[7] == self.board[8] == player) or 
                (self.board[0] == self.board[3] == self.board[6] == player) or 
                (self.board[1] == self.board[4] == self.board[7] == player) or 
                (self.board[2] == self.board[5] == self.board[8] == player) or 
                (self.board[0] == self.board[4] == self.board[8] == player) or 
                (self.board[2] == self.board[4] == self.board[6] == player)):
                result = player

        # If no player won and there are no more legal moves, the game has ended in a draw
        if result == None and len(self.legal_moves) == 0:
            result = 'draw'
        
        # Return the encoded result of the game as 1 (x won), -1 (o won), 0 (game ended in a draw) or None (game still in progress)
        if result == 'x':
            return 1
        elif result == 'o':
            return -1
        elif result == 'draw':
            return 0
        else:
            return None


    '''Returns the result of the game'''
    def get_result(self):
        return self.result


    '''Returns a list of all legal moves in the current position'''
    def get_legal_moves(self):
        return self.legal_moves


    '''Returns the last move of the game'''
    def get_last_move(self):
        # If the game history is empty return None, else return the last move
        return self.game_history[-1] if len(self.game_history) > 0 else None
    

    '''Returns the history of the current game'''
    def get_game_history(self, event, round, player1, player2):
        return str(self.game_history) + '\n'
        
    
    '''Returns the move temperature based on the number of moves played in the current game'''
    def get_move_temperature(self):
        # If the number of moves played in the current game is smaller or equal to the move threshold, return tau
        if len(self.game_history) <= self.move_threshold:
            return self.tau
        # Else return None, which leads to applying the argmax operator and thus picking the most visited node
        else:
            return None