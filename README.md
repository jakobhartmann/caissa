# Caissa

## Introduction
Caissa is an algorithm which I developed as part of my bachelor thesis "Learning Three-Check Chess From Self-Play Using Deep Reinforcement Learning" at TU Berlin in 2021. It is inspired by the success of DeepMind's AlphaZero algorithm and based on the paper

> D. Silver _et al._, “[A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://doi.org/10.1126/science.aar6404),” _Science_, vol. 362, no. 6419, pp. 1140–1144, Dec. 2018.

Like AlphaZero, Caissa is in principle capable of learning any two-player, zero-sum, perfect information game just by knowing the rules of the game and playing training games against itself. For this purpose, the basic algorithm was separated from the game-specific implementation. This allows new games to be added quickly and new features of the algorithm to be tested with games where a ground truth exists. At the moment two games are available: Tic-Tac-Toe and Three-Check Chess. The latter has been implemented with the help of the [python-chess](https://github.com/niklasf/python-chess) library.

## Installation
- Start by cloning the repository:
```
git clone https://github.com/jakobhartmann/caissa.git
```
- The algorithm requires Python together with the following packages: `numpy`, `matplotlib`, `scikit-learn`, `tensorflow` and `chess`. 
- To batch the inference requests to the neural network, [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) and the `tensorflow-serving-api` have to be installed. If you only want to use the search algorithm, TensorFlow Serving could theoretically be omitted, but this would significantly slow down the search process. When starting the TensorFlow Serving Docker container, please make sure that batching is enabled, otherwise the tree parallelization and parallel game generation features will be useless.
- If a GPU is available, `tensorflow-gpu`, `tensorflow-serving-api-gpu` and [TensorFlow Serving GPU](https://www.tensorflow.org/tfx/serving/docker#serving_with_docker_using_your_gpu) should be used. 
- To switch between different games, just change the import statements in the Python files. Each game has its own configuration file which contains all relevant (hyper-)parameters used for search and training.

## Usage
Below is a list of some of Caissa's features and how to use them. The program code has been thoroughly annotated, please read through the comments if any questions arise.

- **MCTS search**
```python
transposition_table = TranspositionTable()
mcts = MonteCarloTreeSearch(transposition_table = transposition_table, model = None)
board = GameBoard()
root = mcts.search(board = board, num_iterations = 800, root = None)
child, move = mcts.move_selection(root = root, exploratory = False)
```
Please note that if `MonteCarloTreeSearch` is passed a model at instantiation, it will be used for the inference requests, otherwise the requests will be made using TensorFlow Serving.

- **Start training**
```python
neural_network = NeuralNetwork()
model = neural_network.get_model()
model.save(Config.MODEL_BACKUP_DIRECTORY + '0')
time.sleep(Config.WAIT_TIME_UNTIL_NEXT_MODEL_ITERATION)
training = Training(model = model)
training.run()
```

- **Resume training**
```python
training = Training(model = None)
training.resume_training(num_model_iterations = 10, first_model_iteration = 5, continue_last_model_iteration = False)
```
Set `continue_last_model_iteration` to `True` if the iteration was interrupted during the game generation phase and to `False` if the iteration should be started from scratch.

- **Plot training progress**
```python
training.plot_history(history = training.training_history, x_label = 'epoch')
training.plot_history(history = training.aggregated_training_history, x_label = 'model iteration')
```

## Contribution

If you are interested in contributing - there are several ways in which Caissa can be improved:
1. The greatest strength of AlphaZero-like algorithms, learning a game through self-play without having any domain specific knowledge, is also its greatest weakness: Generating training data through self-play takes a long time and is computationally expensive. The most important features to speed up the training process are already implemented: Tree paralellization through multithreading and parallel game generation through multiprocessing. However, there is still room for improvement, e.g. during the selection phase of Monte Carlo Tree Search (MCTS) the PUCT score has to be calculated only for a subset of the child nodes, currently it is calculated for all of them.
2. If you have a computer or server with a strong CPU and GPU, you can continue the training process for Three-Check Chess using the latest network. The playing strength of the algorithm will increase with the number of training iterations.
3. New games such as Connect Four or Go could be added. The necessary functions that the algorithm needs can be derived from the games that have already been implemented.
4. I'm sure there are many more possibilities for improving Caissa, I look forward to your contributions!


## License
Caissa is licensed under the GPL 3 (or any later version at your option). Check out LICENSE.txt for the full text.