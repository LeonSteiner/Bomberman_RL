import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from .myfuncs import DQN


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = DQN(15, 15, 6)
        print('new model')
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        print('load model')
    self.inv_num = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    self.logger.debug("Querying model for action.")

    if self.train:
        # return a random action
        if random.random() < self.exploration_rate:
            self.logger.debug("Choosing action purely at random.")
            return np.random.choice(ACTIONS, p=[.22, .22, .22, .22, .12, 0.])
        # make policy net choose an action
        else:
            with torch.no_grad():
                myprobs = self.policy_net(state_to_features(game_state).view(1,1,15,15)).detach().numpy()[0]
            myprobs = myprobs[:-1]
            self.Q_list.append(np.max(myprobs))
            return ACTIONS[np.argmax(myprobs)]
    else:
        # choose only best action when not training
        with torch.no_grad():
            myprobs = self.model(state_to_features(game_state).view(1,1,15,15)).detach().numpy()[0]
        myprobs = myprobs[:-1]
        return ACTIONS[np.argmax(myprobs)] 
    
    


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    full_field = np.copy(game_state['field'])
    coin_locations = np.array(game_state['coins']).T.tolist()
    full_field[tuple(coin_locations)] = 2 # coins have value 2 on board
    full_field[game_state['self'][-1]] = 100 # own position has value 3
    # now crop the outer borders bc they are always the same
    full_field = full_field[1:-1, 1:-1]
    return torch.from_numpy(full_field).float()
