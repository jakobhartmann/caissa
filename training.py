import os
import time
import pickle
import threading
from multiprocessing import Queue
import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from neural_network import NeuralNetwork
from training_set import TrainingSet
from self_play import SelfPlay

# from tic_tac_toe.config import Config
# from tic_tac_toe.transposition_table import TranspositionTable

from three_check_chess.config import Config
from three_check_chess.transposition_table import TranspositionTable


'''Handles the communication between the different processes generating training games and updates the model with these games'''
class Training:
    '''Initialize the class variables'''
    def __init__(self, model):
        # Number of the first model iteration
        self.first_model_iteration = 1
        # Number of training iterations
        self.num_model_iterations = Config.NUM_MODEL_ITERATIONS
        # Number of MCTS iterations used to determine the next move
        self.num_mcts_iterations = Config.NUM_MCTS_ITERATIONS
        # Dictionary with the number of training games generated in the corresponding iteration
        self.num_games = Config.NUM_GAMES
        # Neural network class giving access to the custom loss functions
        self.neural_network = NeuralNetwork()
        # Current version of the model used for inference and training
        self.model = model
        # Batch size used for training
        self.batch_size = Config.BATCH_SIZE
        # Number of epochs used for training
        self.num_epochs = Config.NUM_EPOCHS
        # Dictionary with training iterations and their respective learning rates
        self.learning_rates = Config.LEARNING_RATES
        # Nesterov momentum for SGD
        self.momentum = Config.MOMENTUM
        # Percentage of training data used for the validation set
        self.validation_split = Config.VALIDATION_SPLIT
        # Complete training history for each epoch
        self.training_history = {}
        # Complete aggregated training history for each model iteration
        self.aggregated_training_history = {}
        # Number of previous iterations from which games should be saved and used for model training
        self.num_saved_previous_iterations = Config.NUM_SAVED_PREVIOUS_ITERATIONS
        # Class to store and update the training set
        self.training_set = TrainingSet()
        # Number of games the training data is periodically saved to disk
        self.num_games_training_data_backup = Config.NUM_GAMES_TRAINING_DATA_BACKUP
        # Number of processes which are simultaneously generating training games
        self.num_processes = Config.NUM_PROCESSES
        # List of processes that have finished execution, prevents data being send to dead processes
        self.finished_processes = []
        # Queue used by the processes to send training games and the corresponding game boards
        self.training_games_queue = Queue()
        # Main transposition table that can be periodically updated by the transposition tables of the different processes, if this feature is activated
        self.transposition_table = None
        # The number of games the transposition table is periodically sent to the main process for updates (a negative number deactivates the feature)
        self.num_games_transposition_table_update = Config.NUM_GAMES_TRANSPOSITION_TABLE_UPDATE
        # Whether or not the transposition table should be saved to disk
        self.save_transposition_table = Config.SAVE_TRANSPOSITION_TABLE
        # Queue used by the processes to periodically send their transposition table
        self.get_transposition_table_queue = None
        # List of queues used by the main process to send updated transposition tables
        self.put_transposition_table_queues = [None for _ in range(self.num_processes)]
        # Time in seconds to wait until the next training iteration so that TensorFlow Serving can load the new model
        self.wait_time_until_next_model_iteration = Config.WAIT_TIME_UNTIL_NEXT_MODEL_ITERATION
        # Backup directory to save the models for TensorFlow Serving
        self.model_backup_directory = Config.MODEL_BACKUP_DIRECTORY
        # Backup directory to save the training games, game histories, transposition table and training history
        self.data_backup_directory = Config.DATA_BACKUP_DIRECTORY
        # File extension for saving the game histories
        self.game_histories_file_extension = Config.GAME_HISTORIES_FILE_EXTENSION


    '''Resumes the training from the given starting point for the given number of iterations and handles the relevant preprocessing'''
    def resume_training(self, num_model_iterations, first_model_iteration, continue_last_model_iteration):
        # Number of training iterations
        self.num_model_iterations = num_model_iterations
        # Training iteration at which the training should be continued
        self.first_model_iteration = first_model_iteration

        # Load the last model
        self.model = keras.models.load_model(self.model_backup_directory + str(self.first_model_iteration - 1), custom_objects = {'mse_loss': self.neural_network.mse_loss, 'ce_loss': self.neural_network.ce_loss})

        # Iterate over all previous iterations
        for model_iteration in range(1, self.first_model_iteration):
            # Create the name of the directory
            dirname = self.data_backup_directory + str(model_iteration)

            # Load the respective training history and process it
            infile = open(dirname + '/history', 'rb')
            previous_history = pickle.load(infile)
            infile.close()
            self.process_history(previous_history, False)

            # If this iteration is relevant for future training, load the respective training games and add them to the training set
            if model_iteration >= self.first_model_iteration - self.num_saved_previous_iterations:
                infile = open(dirname + '/training_games', 'rb')
                previous_training_games = pickle.load(infile)
                infile.close()
                self.training_set.training_games_previous_iterations += previous_training_games

        # If the training of an already started iteration should be continued, load the generated training games together with their corresponding game histories and add them to the training set
        if continue_last_model_iteration:
            # Create the name of the directory
            dirname = self.data_backup_directory + str(self.first_model_iteration)

            # Load and save the training games
            infile = open(dirname + '/training_games', 'rb')
            current_training_games = pickle.load(infile)
            infile.close()
            self.training_set.training_games_current_iteration += current_training_games

            # Load and save the game histories
            infile = open(dirname + '/games' + self.game_histories_file_extension, 'r')
            for entry in infile:
                self.training_set.game_histories.append(entry)
            infile.close()

        # Continue training
        self.run()

    
    '''Main function that performs the training'''
    def run(self):
        # For the given number of iterations
        for self.model_iteration in range(self.first_model_iteration, self.num_model_iterations + self.first_model_iteration):
            print('Model iteration ' + str(self.model_iteration) + ' started.')
            # Generate the training games
            self.generate_training_data()
            # Get the preprocessed training data
            bitboard_train_tensor, policy_train_tensor, value_train_tensor = self.training_set.get_training_data()
            # If it is the first model iteration or the learning rate is updated, initialize the optimizer and compile the model
            if self.model_iteration in self.learning_rates.keys():
                sgd_optimizer = tf.keras.optimizers.SGD(learning_rate = self.learning_rate_scheduler(), momentum = self.momentum, nesterov = True)
                self.model.compile(loss = [self.neural_network.ce_loss, self.neural_network.mse_loss], optimizer = sgd_optimizer)
            # Train the model
            history = self.model.fit(x = bitboard_train_tensor, y = [policy_train_tensor, value_train_tensor], batch_size = self.batch_size, epochs = self.num_epochs, verbose = 2, validation_split = self.validation_split)
            # Save and process the training history
            self.process_history(history.history, True)
            # Save the model for tensorflow serving
            self.model.save(self.model_backup_directory + str(self.model_iteration))
            # Update the training set for the next iteration
            self.training_set.update_training_set(self.model_iteration)
            # If this was not the last iteration, wait for TensorFlow Serving to load the new model
            if self.model_iteration != self.num_model_iterations + self.first_model_iteration - 1:
                time.sleep(self.wait_time_until_next_model_iteration)

    
    '''Spawns and handles the processes which generate the training games'''
    def generate_training_data(self):
        # If the folder for saving the data of this model iteration does not already exist, create it
        dirname = self.data_backup_directory + str(self.model_iteration)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Initialize a new transposition table
        # Important!!! Always put a deepcopy of the transposition table into the queues to prevent errors and process crashes!!!
        self.transposition_table = TranspositionTable()

        # In case the training of an already started training iteration is continued, substract the number of the already generated training games from the total number of games to generate
        num_games_to_generate = self.num_games[self.model_iteration] - len(self.training_set.training_games_current_iteration)

        # Distribute the number of games as evenly a possible among the number of processes
        num_games_per_process = [num_games_to_generate // self.num_processes for _ in range(self.num_processes)]
        modulo = num_games_to_generate % self.num_processes
        while modulo > 0:
            num_games_per_process[modulo] += 1
            modulo -= 1
        # List to store all processes
        processes = []

        # Initialize and start a new thread that saves the incoming training games
        training_games_thread = threading.Thread(target = self.save_training_games)
        training_games_thread.start()
        # If the update feature for the transposition table is activated, initialize and start a new thread that updates the main transposition table with the incoming ones and distributes it back to the processes
        if self.num_games_transposition_table_update > 0:
            self.get_transposition_table_queue = Queue()
            self.put_transposition_table_queues = [Queue() for _ in range(self.num_processes)]
            transposition_table_thread = threading.Thread(target = self.update_transposition_table)
            transposition_table_thread.start()

        # Initialize and start each process and save it to the list
        for i in range(self.num_processes):
            p = SelfPlay(self.model_iteration, i, num_games_per_process[i], self.training_games_queue, self.get_transposition_table_queue, self.put_transposition_table_queues[i], copy.deepcopy(self.transposition_table))
            p.start()
            processes.append(p)
        # Join each process to wait for the execution to finish
        for p in processes:
            p.join()

        # Put a sentinel object (None) in the queues to signal the threads that the game generation phase has ended and wait for them to finish their work
        self.training_games_queue.put((None, None))
        training_games_thread.join()
        if self.num_games_transposition_table_update > 0:
            self.get_transposition_table_queue.put(None)
            transposition_table_thread.join()
        
    
    '''Saves the incoming training games to the training set and periodically also to disk'''
    def save_training_games(self):
        # Stay in the loop until a sentinel object is detected
        while True:
            # Wait until a new object is send to the queue and then unpack it
            game, board = self.training_games_queue.get()
            # If the object is not a sentinel object, save the training game and its history to the training set and periodically also to disk
            if game != None:
                self.training_set.training_games_current_iteration.append(game)
                num_played_games = len(self.training_set.training_games_current_iteration)
                self.training_set.game_histories.append(board.get_game_history(event = 'Training Game', round = num_played_games, player1 = 'Model Iteration ' + str(self.model_iteration), player2 = 'Model Iteration ' + str(self.model_iteration)))
                if num_played_games % self.num_games_training_data_backup == 0:
                    self.save_file(self.training_set.training_games_current_iteration, '/training_games', 'wb')
                    self.save_file(self.training_set.game_histories, '/games' + self.game_histories_file_extension, 'w')
                    print(str(num_played_games) + ' training games have been played and saved to disk.')
            # If the object is a sentinel object, save the training games and game histories to disk and quit the loop
            else:
                self.save_file(self.training_set.training_games_current_iteration, '/training_games', 'wb')
                self.save_file(self.training_set.game_histories, '/games' + self.game_histories_file_extension, 'w')
                break


    '''Updates the main transposition table with the incoming ones and distributes it back to the processes'''
    def update_transposition_table(self):
        # Stay in the loop until a sentinel object is detected
        while True:
            # Wait until a new object is send to the queue and then unpack it
            hash_table = self.get_transposition_table_queue.get()
            # If the object is an integer, add the process id to the list of finished processes and close the corresponding queue
            if isinstance(hash_table, int):
                process_id = hash_table
                self.finished_processes.append(process_id)
                self.put_transposition_table_queues[process_id].close()
            # If the object is a sentinel object, save the transposition table to disk, close the get_transposition_table_queue, reinitialize the list of finished processes for the next iteration and quit the loop
            elif hash_table == None:
                if self.save_transposition_table:
                    self.save_file(self.transposition_table.hash_table, '/transposition_table', 'wb')
                self.get_transposition_table_queue.close()
                self.finished_processes = []
                break
            # Else, if the object is a transposition table/hash table, use it to update the main transposition table, distribute the updated version back to the processes which are still running and save it to disk
            else:
                self.transposition_table.hash_table.update(hash_table)
                for i in range(self.num_processes):
                    if i not in self.finished_processes:
                        self.put_transposition_table_queues[i].put(copy.deepcopy(self.transposition_table.hash_table))
                if self.save_transposition_table:
                    self.save_file(self.transposition_table.hash_table, '/transposition_table', 'wb')
                

    '''Pickles and saves the given file to disk'''
    def save_file(self, file, filename, mode):
        # Create the name of the directory
        dirname = self.data_backup_directory + str(self.model_iteration)
        # If the folder does not already exist, create it
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # If the file does already exist, delete it
        if os.path.exists(dirname + filename):
            os.remove(dirname + filename)
        # Get the output file
        outfile = open(dirname + filename, mode)
        # If the data should be saved in binary mode, pickle the data and write it to the file
        if mode == 'wb':
            pickle.dump(file, outfile)
        # Else should the data be saved in text mode, iterate over the list and write each entry to the file
        elif mode == 'w':
            for entry in file:
                outfile.write(entry)
        # Close the output file
        outfile.close()


    '''Returns the learning rate based on the current model iteration'''
    def learning_rate_scheduler(self):
        return self.learning_rates[self.model_iteration]


    '''Adds the history of one training iteration to the complete (aggregated) training history and if wanted also saves it to disk'''
    def process_history(self, history, save):
        # If indicated, save the training history
        if save:
            self.save_file(history, '/history', 'wb')

        # Add the history of this iteration to the complete training history
        for key, value in history.items():
            if key in self.training_history:
                self.training_history[key].extend(value)
            else:
                self.training_history[key] = value

        # Add the aggregated history of this iteration to the complete aggregated training history
        for key, value in history.items():
            if key in self.aggregated_training_history:
                self.aggregated_training_history[key].append(np.mean(value))
            else:
                self.aggregated_training_history[key] = [np.mean(value)]


    '''Plots the training history'''
    def plot_history(self, history, x_label = 'epoch'):
        # Get the metrics
        metrics = list(history.keys())
        train_loss = metrics[0]
        train_policy_loss = metrics[1]
        train_value_loss = metrics[2]
        val_loss = metrics[3]
        val_policy_loss = metrics[4]
        val_value_loss = metrics[5]
        # Define x axis
        x_axis = [i for i in range(1, len(history[train_loss]) + 1)]
        # Create a plot with tree subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 4), sharey = False)
        # First subplot: model loss
        ax1.set_title('model loss')
        ax1.set(ylabel = 'loss')
        ax1.plot(x_axis, history[train_loss], label = 'train')
        ax1.plot(x_axis, history[val_loss], label = 'validation')
        ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
        # Second subplot: policy loss
        ax2.set_title('policy loss')
        ax2.set(xlabel = x_label)
        ax2.plot(x_axis, history[train_policy_loss], label = 'train')
        ax2.plot(x_axis, history[val_policy_loss], label = 'validation')
        ax2.xaxis.set_major_locator(MaxNLocator(integer = True))
        # Third subplot: value loss
        ax3.set_title('value loss')
        ax3.plot(x_axis, history[train_value_loss], label = 'train')
        ax3.plot(x_axis, history[val_value_loss], label = 'validation')
        ax3.xaxis.set_major_locator(MaxNLocator(integer = True))
        # Position the legend
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc = 'center right')
        # Show the plot
        plt.show()