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



################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("No executable code, meant for use as module only")