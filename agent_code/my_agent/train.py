import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from .myfuncs import *

# new
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 300000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
RANDOM_BATCH_SIZE = 64
GAMMA = 0.99
TAU = 5e-4 # update rate for target network
LEARNING_RATE = 5e-4
UPDATE_EVERY = 10

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

'''def action2number(actionstr):
    if actionstr == 'UP':
        return 0
    if actionstr == 'RIGHT':
        return 1
    if actionstr =='DOWN':
        return 2
    if actionstr =='LEFT':
        return 3
    if actionstr =='WAIT':
        return 4
    if actionstr =='BOMB':
        return 5'''
ActionDict = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.policy_net = self.model
    self.target_net = DQN(15, 15, 6)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    #self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.01, momentum=0.)
    #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    #self.optimizer = self.loaded_optimizer
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    if not self.resume_training:
        self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
        self.total_steps = 0
    else:
        self.optimizer.load_state_dict(self.optimizer_dummy)

    self.loss_list = []
    self.Q_list = []

    self.exploration_rate = 1

    self.total_loss = 0

    self.all_actions = np.zeros(6)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    # Idea: Add your own events to hand out rewards

    self.total_steps += 1

    events.extend(ev for ev in custom_events(self, old_game_state, new_game_state))

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    if old_game_state != None:
        self.transitions.append(Transition(state_to_features(old_game_state), ActionDict[self_action], state_to_features(new_game_state), reward_from_events(self, events)))

    '''
    if self.total_steps > 2:
        batch = self.transitions[-1]

        state_batch = batch.state
        next_state_batch = batch.next_state
        action_batch = torch.tensor(batch.action) 
        reward_batch = torch.tensor(batch.reward) 

        state_action_value = self.policy_net(state_batch.view(1,1,15,15))[0].gather(0, action_batch)
        next_state_value = self.target_net(next_state_batch.view(1,1,15,15)).max(1)[0].detach()

        expected_state_action_value = reward_from_events(self, events) + next_state_value*GAMMA        
        
        loss = F.smooth_l1_loss(state_action_value.view(1,1), expected_state_action_value.unsqueeze(1)) #vielleicht mse?

        # only update the network every 10th step for stability reasons
        self.total_loss += loss
        if self.total_steps%10 == 0:

            self.optimizer.zero_grad()
            self.total_loss.backward()
            #for param in self.policy_net.parameters():
            #    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
    
            self.loss_list.append(self.total_loss.detach().item())
            self.total_loss = 0'''

    if self.total_steps > 20000 and self.total_steps%UPDATE_EVERY==0:
        # get a random batch from transition buffer
        random_indices = np.random.permutation(np.arange(len(self.transitions)))[:RANDOM_BATCH_SIZE]
        random_batch = [self.transitions[i] for i in random_indices]
        batch = Transition(*zip(*random_batch))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)

        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
        try:
            action_batch = torch.from_numpy(np.asarray(batch.action)).view(RANDOM_BATCH_SIZE, 1)
        except:
            print(batch.action)
        reward_batch = torch.from_numpy(np.asarray(batch.reward))

        # get Q values
        state_action_value = self.policy_net(state_batch.view(RANDOM_BATCH_SIZE,1,15,15)).gather(1, action_batch)
        next_state_value = torch.zeros(RANDOM_BATCH_SIZE)
        next_state_value[non_final_mask] = self.target_net(next_state_batch.view(torch.sum(non_final_mask),1,15,15)).max(1)[0].detach()

        expected_state_action_value = reward_batch + next_state_value*GAMMA        

        # compute loss and do omtimization
        loss = F.smooth_l1_loss(state_action_value, expected_state_action_value.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # append to loss list
        self.loss_list.append(loss.detach().item())

        # do soft target_net update
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #print(last_game_state['step'])
    self.transitions.append(Transition(state_to_features(last_game_state), ActionDict[last_action], None, reward_from_events(self, events)))
    
    if self.total_steps%400 == 0:
        with open("transition_history.pt", "wb") as file:
            pickle.dump(self.transitions, file)
    # Store the model
    # copy the policy network to the target network
    if last_game_state['round']%15 == 0 and self.total_steps > 20000:
        # changed to soft update after optimization
        #self.target_net.load_state_dict(self.policy_net.state_dict())
            
        
        torch.save({
            'epoch': last_game_state['round'],
            'total_steps':self.total_steps,
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, 'my-saved-model.tar')

        print('policy network model saved')
            torch.save({
            'epoch': last_game_state['round'],
            'total_steps':self.total_steps,
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, 'checkpoints/my-saved-model{}.tar'.format(last_game_state['round']))
        if last_game_state['round']%120 == 0:

        # save loss for analysis
        np.save('metrics/loss{}'.format(last_game_state['round']), np.array(self.loss_list))
        np.save('metrics/Q{}'.format(last_game_state['round']), np.array(self.Q_list))
        self.loss_list = []
        self.Q_list = []

    # print points
        print(self.all_actions, 'inv', self.inv_num, '\tpoints',
            last_game_state['self'][1], 'expl', np.round(self.exploration_rate, 3))
    self.inv_num = 0

    # decrease exploration rate after each game
    if self.total_steps > 20000:
        self.exploration_rate = max(0.1, self.exploration_rate*0.999)
    #print(self.all_actions, self.init_pos, np.round(self.exploration_rate, 3))
    self.all_actions = np.zeros(6)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        #e.KILLED_OPPONENT: 50,
        #e.MOVED_DOWN: 1,
        #e.MOVED_LEFT: 0,
        #e.MOVED_RIGHT: 0,
        #e.MOVED_UP: 1,
        e.WAITED: -1,
        e.INVALID_ACTION: -2,
        e.MOVED_TO_COIN: 3,
        e.MOVED_AWAY_FROM_COIN: -2,
        e.GOT_KILLED: -20,
        e.MOVED_TO_BOMB: -5,
        e.MOVED_AWAY_FROM_BOMB: 5,
        e.BOMB_DROPPED: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if event=='INVALID_ACTION':
            self.inv_num += 1

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
