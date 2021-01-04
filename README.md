# AlphaZero

## Introduction
Implementation of AlphaZero following the [original paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).

AlphaZero is a self-play reinforcement learning algorithm originally devised to master Go game, but has proven to
be a very good approximation also for chess and, in general, zero-sum games of perfect information. 

AlphaZero employs a Monte Carlo Tree Search (MCTS) to sample the action tree of a game from the current state, guided 
by a deep neural network (DNN) to evaluate prior probabilities of each action and a value estimating the probability of 
the player winning the game.

Given a root node a MCTS self-play consist on the following steps:

1. Select:
Starting from root node (s0), legal actions are performed until a leaf node in found 
(a node not yet explored or game ending). Among the possible valid actions, the one 
that maximizes the node value plus prior probability is selected.

2. Expand:
Once the simulation reaches a leaf node for which no prior probabilities are known,
it is expanded. For that, the state is evaluated with the DNN and prior probabilities 
are assigned to each of the children nodes.

3. Backup:
The nodes visited to reach the leaf node increment their visit count by one unit, and the 
values evaluated in the leaf nodes are backpropagated. Each node's value is the mean of 
the values of the children values.
This three steps are carried for a number of games so statistics on each node are generated.
The most promising actions from a current state are those whose children nodes are more visited.

4. Play:
Once the statistics of each node are populated an action to be played from root node is selected.
The probability of an action being played is directly related to the number of times this child node
was visited during the exploration process.

Self-plays are saved to be used as dataset to train the DNN predict better policies and values for the
nodes, leading to better exploration processes during the MCTS.
## Contents:
This repository contains the following files:

- MCTS.py: MCTS class. Holds all the operations that are run in the MCTS process.

- Node.py: Node class. 

- ReplayMemory.py: On this object the games are saved to be used as dataset in the training process.

- Trainer.py: Class holding the methods to train the DNN.

- Tournament.py: Function to make plays between two agents. The winning statistics are generated for
each agent.

- Play.py: Run it inside a game folder to play against the best agent.

- train.py: Executes the training process. It should be run from a game folder.

Game dependent content:

- Game.py. Game class. The following methods must be implemented:

1. `play(node, action)`: Given the `state` (in node) of the game and `action` is played by `player`. The
new state is returned.

2. `check_end(node)`: Given `state` (in node) returns `end, winner`, with `end` being a bool indicating if `state`
corresponds to end game and `winner` is the index of player who won (0 if draw).

3. `avail_actions(node)`: Given `state`  (in node) returns an array of possible actions to be played.

4. `plot(node)`: Plots the current state in a human easy interpretable way.

The states are always from the side of the current player.

- DNN.py: The deep neural network function. Since I do not have the computer resources of Google I did not
implement the net described in the article but a much simpler one...

 
## Training process:

1. Game and MCTS objects are instantiated. Two players are generated (at this point they are the same)
current and best player.

Training loop:

2. `GAMES_PER_EPISODE` games of self-play are run to populate the replay memory. The best player
is used to evaluate prior probabilities.

3. Current player is trained over the set of data generated.

4. If `EPISODE == TOURNAMENT_FREQ` a number of games are played between the current 
agent and the best agent. The current agent becomes the best if the former wins more 
than 50 % of the games.

In the original paper this three process run in parallel. Instead, a sequential implementation is used 
here.

To run the training process, go inside a game folder. From there execute
`
python ../train.py
`
with the following options:

```
usage: train.py [-h] [--episodes EPISODES] [--ngames NGAMES]
                [--games_per_episode GAMES_PER_EPISODE]
                [--tournament_freq TOURNAMENT_FREQ] [--batch_size BATCH_SIZE]
                [--nbatch NBATCH] [--nepochs NEPOCHS] [--lr LR] [--mom MOM]
                [--capacity CAPACITY] [--replay_start_size REPLAY_START_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --episodes EPISODES   Number of episodes in training process.
  --ngames NGAMES       Number of self-play games in each MCTS.
  --games_per_episode GAMES_PER_EPISODE
                        Number of self-play games before each training
                        episode.
  --tournament_freq TOURNAMENT_FREQ
                        Frequency, in episodes, to make tournament between
                        current player and best player.
  --batch_size BATCH_SIZE
                        Batch size.
  --nbatch NBATCH       Number of batches at each epoch.
  --nepochs NEPOCHS     Number of epochs
  --lr LR               learning rate
  --mom MOM             momentum
  --capacity CAPACITY   Maximum number of datapoints in replaymemory.
  --replay_start_size REPLAY_START_SIZE
                        Minimum number of datapoint to start training.
```
## References:
- [Original paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)

- [MCTS implementation](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/)

- [Pseudocode](https://github.com/jianpingliu/AlphaZero/)

- [Connect four](https://mathworld.wolfram.com/Connect-Four.html)
