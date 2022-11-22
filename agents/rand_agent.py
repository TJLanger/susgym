#!/usr/bin/env python

################################################################################
################################################################################
## Authorship
################################################################################
################################################################################
##### Project
############### susgym
##### Name
############### rand_agent.py
##### Author
############### Trenton Langer
##### Creation Date
############### 20221121
##### Description
############### Stochastic agents for use in an OpenAI gym based environment
##### Project
############### YYYYMMDD - (CONTRIBUTOR) EXAMPLE



################################################################################
################################################################################
## Imports
################################################################################
################################################################################

# Other
import numpy as np



################################################################################
################################################################################
## Subroutines
################################################################################
################################################################################



################################################################################
################################################################################
## Classes
################################################################################
################################################################################
"""
description:
-> Fully stochastic agent, usable in any environment. Simply samples from act space
"""
class randAgent():
    def __init__(self):
        self.reward = 0

    def pick_action(self, state, act_space, obs_space):
        action = act_space.sample()
        return action

    def update(self, next_state, reward, done, info):
        self.reward += reward

    def getReward(self):
        return self.reward

    def reset(self):
        self.reward = 0

"""
description:
-> Stochastic agent, with game knowledge to make random (but valid) action choices
"""
class randSusAgent():
    def __init__(self, num_players):
        self.reward = 0
        self.num_players = num_players
        self.num_characters = 10

    def pick_action(self, state, act_space, obs_space):
        # Setup
        act_idx = 0
        action = np.zeros(act_space.shape, dtype=np.int8)
        # State Decode
        state_idx = 1 # skip invite deck index
        bank_gems = state[state_idx:state_idx+3]
        for player_idx in range(self.num_players+1):
            state_idx += 3 # increment past gem counts for bank and all players
        char_locs = state[state_idx:state_idx+2*self.num_characters]
        state_idx += 2*self.num_characters
        die_rolls = state[state_idx:state_idx+2]
        state_idx += 2
        room_gems = state[state_idx:state_idx+3]
        state_idx += 3
        act_cards = []
        for card in range(2):
            act_cards.append(state[state_idx:state_idx+16])
            state_idx += 16
        # Action Creation
        die_rolls = list(map(lambda x: np.random.randint(0, self.num_characters) if x == self.num_characters else x, list(die_rolls)))
        for roll in die_rolls:
            action[act_idx] = roll
            curr_loc = char_locs[2*roll:2*roll+1]
            action[act_idx+1:act_idx+2] = curr_loc
            act_idx += 3
        # Return
        return action

    def update(self, next_state, reward, done, info):
        self.reward += reward

    def getReward(self):
        return self.reward

    def reset(self):
        self.reward = 0



################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("No executable code, meant for use as module only")
