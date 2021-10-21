'''Stores information about one game state'''
class GameState:
    '''Initialize the class variables'''
    def __init__(self, bitboard, policy, value = None):
        # Bitboard of the board
        self.bitboard = bitboard
        # Move probabilities returned by MCTS
        self.policy = policy
        # Final result of the game from the perspective of the side to move (added after the game has ended)
        self.value = value

    '''String representation of a game state'''
    def __str__(self):
        result = ''
        result += 'Bitboard: ' + repr(self.bitboard) + '\n'
        result += 'Policy: ' + repr(self.policy) + '\n'
        result += 'Value: ' + str(self.value) + '\n'
        return result