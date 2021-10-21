from multiprocessing import Process
import copy
import pickle

from monte_carlo_tree_search import MonteCarloTreeSearch
from game_state import GameState

# from tic_tac_toe.config import Config
# from tic_tac_toe.game_board import GameBoard
# from tic_tac_toe.board_utils import BoardUtils

from three_check_chess.config import Config
from three_check_chess.game_board import GameBoard
from three_check_chess.board_utils import BoardUtils


'''Generates training games through self-play in a seperate process'''
class SelfPlay(Process):
    '''Initialize the class variables'''
    def __init__(self, model_iteration, process_id, num_games, training_games_queue, put_transposition_table_queue, get_transposition_table_queue, transposition_table):
        # Call the instructor of the parent class
        Process.__init__(self)
        # Current model iteration
        self.model_iteration = model_iteration
        # Id to identify the process
        self.process_id = process_id
        # Number of self-play games this process should generate
        self.num_games = num_games
        # Queue to send the completed training games together with their corresponding game boards to the main process
        self.training_games_queue = training_games_queue
        # Queue to periodically send the transposition table to the main process
        self.put_transposition_table_queue = put_transposition_table_queue
        # Queue to get the updated transposition table of the main thread
        self.get_transposition_table_queue = get_transposition_table_queue
        # Transposition table
        self.transposition_table = transposition_table
        # The number of games the transposition table is periodically sent to the main process for updates (a negative number deactivates the feature)
        self.num_games_transposition_table_update = Config.NUM_GAMES_TRANSPOSITION_TABLE_UPDATE
        # Whether or not the transposition table should be saved to disk
        self.save_transposition_table = Config.SAVE_TRANSPOSITION_TABLE
        # Backup directory to save the transposition table
        self.data_backup_directory = Config.DATA_BACKUP_DIRECTORY
        # Number of MCTS iterations used to determine the next move
        self.num_mcts_iterations = Config.NUM_MCTS_ITERATIONS
        # BoardUtils class which can generate bitboards and assign moves to their corresponding policy indices and vice versa
        self.board_utils = BoardUtils()


    '''Main function generating the training games'''
    def run(self):
        # Wrap the self-play in a try-catch block
        try:
            # Create the MCTS instance
            self.mcts = MonteCarloTreeSearch(self.transposition_table, None)

            for i in range(self.num_games):
                # If the update feature for the transposition table is activated
                if self.num_games_transposition_table_update > 0:
                    # Periodically send the transposition table to the main process
                    if i % self.num_games_transposition_table_update == 0 and i > 0:
                        self.put_transposition_table_queue.put(copy.deepcopy(self.transposition_table.hash_table))
                    # Check whether the main process send a new transposition table and if so, update the own one with it
                    while not self.get_transposition_table_queue.empty():
                        self.transposition_table.hash_table.update(self.get_transposition_table_queue.get())

                # Initialize a new board
                board = GameBoard()
                # Set the root to None
                root = None
                # Initialize an empty list to store the game states/moves of the game
                game_states = []

                # While the game is not over
                while board.get_result() == None:
                    # Run MCTS search from the current position and the given number of iterations
                    root = self.mcts.search(board, self.num_mcts_iterations, root)

                    # Get the move probabilites
                    move_probabilities = self.mcts.get_move_probabilities(root)
                    # Pick a move stochastically and get the corresponding node
                    root, move = self.mcts.move_selection(root, True)
                    # Create a new game state with the bitboard representation of the current position and the move probabilities obtained from the MCTS search
                    game_state = GameState(self.board_utils.get_bitboard(board)[0], move_probabilities, None)
                    # Append the game state to the list
                    game_states.append(game_state)
                    # Make the chosen move on the board
                    board.make_move(move)

                # IMPORTANT!!!
                # If the game was decisive, the last state that is stored (which is the penultimate state/position of the game) is always marked with a -1!!!
                turn = -1
                # Add the final result of the game to each game state
                for game_state in game_states:
                    # The result of the game is stored for each game state with respect to the side which made the last move!!!
                    game_state.value = board.get_result() * turn
                    # Negate the turn variable for the next iteration
                    turn *= -1

                # Push the training game to the queue together with the corresponding game board
                self.training_games_queue.put((game_states, board))

            # If the update feature for the transposition table is activated
            if self.num_games_transposition_table_update > 0:
                # Notify the main process, that this process has finished its work and that it should not send any more transposition tables
                self.put_transposition_table_queue.put(self.process_id)
                # Send the last version of the transposition table to the main process
                self.put_transposition_table_queue.put(copy.deepcopy(self.transposition_table.hash_table))
                # Empty the queue before ending the process
                while not self.get_transposition_table_queue.empty():
                    self.get_transposition_table_queue.get()
            
            # If the transposition table should be saved to disk
            if self.save_transposition_table:
                # Create the name of the directory and file
                dirname = self.data_backup_directory + str(self.model_iteration)
                filename = '/transposition_table_' + str(self.process_id)
                # Get the output file
                outfile = open(dirname + filename, 'wb')
                # Pickle the transposition table and write it to the file
                pickle.dump(self.transposition_table.hash_table, outfile)
                # Close the output file
                outfile.close()
            
            print('Process ' + str(self.process_id) + ' has finished its execution.')
        # If an exception occurs, print it together with the process id
        except Exception as e:
            print('The exception in process ' + str(self.process_id) + ' is: ' + str(e))