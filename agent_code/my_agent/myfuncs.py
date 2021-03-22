import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# class for network
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # start with 1 channel (only the map)
        # try 6 filters bc 6 different moves
        # kernel size = 3 bc 3 directions (???)
        # if I want to change padding, I have to change conv2d_size_out, too!!!!
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, bias=False)
        # Normalization for faster convergence?
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(24)
        #nn.maxpool2d???

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 24
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def custom_events(self, old_game_state: dict, new_game_state: dict):
	if old_game_state == None:
		return []
	custom_events_list = []
	# old_coin_locations = old_game_state['coins']
	# give pos. reward for going towards coin and neg. for away
	# the distance is not the euclidian norm but x+y because we cannot moce diagonally
	# this is still not perfect bc of the walls but better than nothing
	# using try as an easy way to prevent old_game_state=None or no coins at all (latter not sure?)
	new_own_location = np.asarray(new_game_state['self'][-1])
	old_own_location = np.asarray(old_game_state['self'][-1])

	# here is a try because this code breaks down if there are less than 1 coins
	try:
		new_coin_locations = np.asarray(new_game_state['coins'])
		old_coin_locations = np.asarray(old_game_state['coins'])

		#new_dist_coins = np.sum(np.abs(new_coin_locations - new_own_location), axis=1)
		old_dist_coins = np.sum(np.abs(old_coin_locations - old_own_location), axis=1)

		old_clostest_coin_indx = np.argmin(old_dist_coins)

		#check if the old clostest coin is still there
		if np.sum((old_coin_locations[old_clostest_coin_indx] == new_coin_locations).all(axis=1)):
			# check if the distance has changed
			old_clostest_coin_dist = old_dist_coins[old_clostest_coin_indx]
			new_clostest_coin_dist = np.sum(np.abs(new_coin_locations[old_clostest_coin_indx] - new_own_location))

			# now make the rewards list
			if old_clostest_coin_dist > new_clostest_coin_dist:
				custom_events_list.append('MOVED_TO_COIN')
			
			elif old_clostest_coin_dist < new_clostest_coin_dist:
				custom_events_list.append('MOVED_AWAY_FROM_COIN')


		# rewards for bombs
		for bomb in old_game_state['bombs']:
			bomb_location = bomb[0]
			bomb_timer = bomb[1]

			#print(bomb_timer)
			#if bomb_timer > 0:
			old_dist_bomb = np.sum(np.abs(bomb_location - old_own_location))
			new_dist_bomb = np.sum(np.abs(bomb_location - new_own_location))
			if old_dist_bomb < 4:
				if new_dist_bomb > old_dist_bomb:
					custom_events_list.append('MOVED_AWAY_FROM_BOMB')
				else:
					custom_events_list.append('MOVED_TO_BOMB')
	except Exception as e:
		print(e)
	return custom_events_list