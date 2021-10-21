import tensorflow as tf

# from tic_tac_toe.config import Config

from three_check_chess.config import Config


'''Stores the training data and logic to preprocess and update it'''
class TrainingSet:
    '''Initialize the class variables'''
    def __init__(self):
        # Number of previous iterations from which training games should be saved
        self.num_saved_previous_iterations = Config.NUM_SAVED_PREVIOUS_ITERATIONS
        # Dictionary with the number of training games generated in the corresponding iteration
        self.num_games = Config.NUM_GAMES
        # List to store training games of previous iterations
        self.training_games_previous_iterations = []
        # List to store training games of current iteration
        self.training_games_current_iteration = []
        # List to store game histories
        self.game_histories = []


    '''Preprocess all saved training games and return the tensors used for training'''
    def get_training_data(self):
        # Add the training games from the current iteration to the ones from the previous iterations
        self.training_games_previous_iterations += self.training_games_current_iteration
        # Empty the list with the training games from the current iteration for the next one
        self.training_games_current_iteration = []
        # Initialize the lists for storing the respective parts of the training data
        bitboard_train_data, policy_train_data, value_train_data = [], [], []

        # For each game state in each game in the training set add the bitboard, policy and value to their respective lists
        for game in self.training_games_previous_iterations:
            for game_state in game:
                bitboard_train_data.append(game_state.bitboard)
                policy_train_data.append(game_state.policy)
                value_train_data.append(game_state.value)

        # Convert the lists to tensors
        bitboard_train_data = tf.convert_to_tensor(bitboard_train_data)
        policy_train_data = tf.convert_to_tensor(policy_train_data)
        value_train_data = tf.convert_to_tensor(value_train_data, dtype = tf.float32)

        # Return the tensors
        return bitboard_train_data, policy_train_data, value_train_data


    '''Updates the training set after each iteration'''    
    def update_training_set(self, model_iteration):
        # Determine the number of training games that should be saved from previous iterations (including the current one)
        num_games_saved_from_previous_iterations = 0
        for iteration in range(model_iteration - self.num_saved_previous_iterations + 1, model_iteration + 1):
            if iteration in self.num_games.keys():
                num_games_saved_from_previous_iterations += self.num_games[iteration]

         # If the number of stored training games is greater than the maximum, remove the games from the oldest iteration
        if len(self.training_games_previous_iterations) > num_games_saved_from_previous_iterations:
            self.training_games_previous_iterations = self.training_games_previous_iterations[-num_games_saved_from_previous_iterations:]
            
        # Clear the list with game histories for the next iteration
        self.game_histories = []