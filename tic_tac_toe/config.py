'''Stores all relevant (hyper-)parameters of the algorithm for Tic-Tac-Toe'''
class Config:
    ############
    ### Game ###
    ############

    # Number of possible legal and illegal moves that the neural network can represent
    NUM_MOVES = 9


    ###########################
    ### Transposition Table ###
    ###########################

    # Seed for the random generator
    RANDOM_GENERATOR_SEED = 42
    # Zobrist key for the initial position
    INITIAL_ZOBRIST_KEY = 0
    # Maximum number of entries in the transposition table
    MAX_NUM_ENTRIES_TRANSPOSITION_TABLE = 10000


    #######################
    ### MCTS Parameters ###
    #######################

    # Number of MCTS iterations used to determine the next move
    NUM_MCTS_ITERATIONS = 800
    # Exploration rates for the PUCT score
    C_BASE = 19652
    C_INIT = 1.25
    # Dirichlet noise which is added to the root node
    ALPHA = 2
    EPSILON = 0.25
    # Move temperature to trade-off exploration and exploitation during move selection
    TAU = 1
    MOVE_THRESHOLD_MU = 9
    MOVE_THRESHOLD_SIGMA = 0
    # Number of threads used for tree parallelization
    NUM_THREADS = 8
    # Virtual loss which is added during tree traversal
    VIRTUAL_LOSS = 1


    ######################
    ### Neural Network ###
    ######################

    # Name of the model
    MODEL_NAME = 'tic_tac_toe'
    # Shape of the neural network input, i.e., the bitboard representation
    INPUT_SHAPE = (3, 3, 3)
    # Number of residual blocks in the body of the neural network
    NUM_RESIDUAL_BLOCKS = 6
    # Number of convolutional filters in the body and first layer of the policy head
    NUM_CONVOLUTIONAL_FILTERS = 32
    # Number of convolutional filters in the policy head, representing the number of output planes
    NUM_POLICY_HEAD_FILTERS = 1
    # Number of convolutional filters in the value head
    NUM_VALUE_HEAD_FILTERS = 1
    # L2 regularization strength for the kernel and bias regularizer in the convoluational and dense layers
    L2_REGULARIZATION_STRENGTH = 0.0001


    ##########################
    ### TensorFlow Serving ###
    ##########################

    # TensorFlow Serving URL for inference requests
    MODEL_URL = 'localhost'
    # TenorFlow Serving REST port for inference requests
    REST_PORT = 8501
    # TenorFlow Serving gRPC port for inference requests
    GRPC_PORT = 8500


    ################
    ### Training ###
    ################

    # Number of training iterations
    NUM_MODEL_ITERATIONS = 15
    # Dictionary with the number of training games generated in the corresponding iteration
    NUM_GAMES = {1: 50, 2: 100, 3: 150, 4: 200, 5: 200, 6: 200, 7: 200, 8: 200, 9: 200, 10: 200, 11: 200, 12: 200, 13: 200, 14: 200, 15: 200}
    # Batch size used for training
    BATCH_SIZE = 32
    # Number of epochs used for training
    NUM_EPOCHS = 10
    # Dictionary with training iterations and their respective learning rates
    LEARNING_RATES = {1: 0.1, 11: 0.01, 31: 0.001, 51: 0.0001}
    # Nesterov momentum for SGD
    MOMENTUM = 0.9
    # Percentage of training data used for the validation set
    VALIDATION_SPLIT = 0.05
    # Number of processes which are simultaneously generating training games
    NUM_PROCESSES = 2
    # Number of previous iterations from which games should be used for model training
    NUM_SAVED_PREVIOUS_ITERATIONS = 3
    # Number of games the training data is periodically saved to disk
    NUM_GAMES_TRAINING_DATA_BACKUP = 10
    # The number of games the transposition table is periodically sent to the main process for updates (a negative number deactivates the feature)
    NUM_GAMES_TRANSPOSITION_TABLE_UPDATE = -1
    # Whether or not the transposition table should be saved to disk
    SAVE_TRANSPOSITION_TABLE = True
    # Time in seconds to wait until the next training iteration so that TensorFlow Serving can load the new model
    WAIT_TIME_UNTIL_NEXT_MODEL_ITERATION = 30
    # Backup directory to save the models for TensorFlow Serving
    MODEL_BACKUP_DIRECTORY = 'models/'
    # Backup directory to save the training games, game histories, transposition table and training history
    DATA_BACKUP_DIRECTORY = 'data/'
    # File extension for saving the game histories
    GAME_HISTORIES_FILE_EXTENSION = '.txt'