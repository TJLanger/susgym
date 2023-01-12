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
############### 20230107 - (TLANGER) Convert to using agent_helpers



################################################################################
################################################################################
## Imports
################################################################################
################################################################################

# Helpers
from . import agent_helpers

# Other
import numpy as np

# Debug
from pprint import pprint


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
        action = np.zeros(act_space.shape, dtype=np.int8)
        # State Decode
        ii, bank_gems, pg, char_locs, die_rolls, room_gems, act_cards, k = agent_helpers.decode_state(state, self.num_players, self.num_characters)
        # Action Creation
        if np.any(bank_gems == 0):
            # Character Identity Guesses
            agent_helpers.randActComp_charGuess(action, self.num_players, self.num_characters)
        else:
            # Character Die Moves
            agent_helpers.randActComp_dieMove(action, die_rolls, char_locs, self.num_characters)
            # Action Card Selection
            act_card_idx = np.random.randint(0, 2)
            act_order = np.random.randint(0, 2)
            agent_helpers.randActComp_actCards(action, self.num_players, state, act_card_idx, act_order, self.num_characters)
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
