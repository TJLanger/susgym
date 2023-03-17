#!/usr/bin/env python

################################################################################
################################################################################
## Authorship
################################################################################
################################################################################
##### Project
############### susgym
##### Name
############### example.py
##### Author
############### Trenton Langer
##### Creation Date
############### 20221117
##### Description
############### Basic gym control loop and testing
##### Project
############### YYYYMMDD - (CONTRIBUTOR) EXAMPLE



################################################################################
################################################################################
## Imports
################################################################################
################################################################################

# Testing
import unittest

# OpenAI gym
import gym
from gym.utils.env_checker import check_env
from suspicion_gym import suspicion_gym
from agents.agent_helpers import decode_state
from agents.rand_agent import randSusAgent

# Other
import argparse
import importlib
import numpy as np
import os
import signal
import time

# Debug
from pprint import pprint



################################################################################
################################################################################
## Global
################################################################################
################################################################################



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



################################################################################
################################################################################
## Tests
################################################################################
################################################################################

"""
description:
-> Environment Testing
"""
class TestEnv(unittest.TestCase):
    # Runs prior to each test
    def setUp(self):
        self.agents = [randSusAgent() for _ in range(4)]
        self.env = gym.make("Suspicion-v1", num_players=len(self.agents), gui_size=None)

    # Test Compatability
    def test_compatability(self):
        check_env(self.env)

    # Test reset
    def test_reset(self):
        # Loop and verify
        for i in range(1000):
            # Call Reset
            state = self.env.reset()
            dstate = decode_state(state)
            # Check Gem Counts
            for gem_cnts in dstate["player_gems"]:
                self.assertTrue(np.sum(gem_cnts) == 0)
            # Check Character Positions
            for char_loc in dstate["character_locations"]:
                self.assertTrue(char_loc[0] == 0 or char_loc[0] == 3 or char_loc[1] == 0 or char_loc[1] == 2)

    # Test Step
    def test_step(self):
        self.env.__partialReward = False
        self.env.reset()
        # Test Invalid Action Format
        act1 = np.array([])
        try:
            self.env.step(act1)
            self.assertTrue(False) # Fails test, unless step catches invalid format
        except Exception:
            self.assertTrue(True)
        # Test Invalid Action
        mod = False
        while mod == False:
            agent_idx, state = self.env.observe()
            dstate = decode_state(state)
            pchar = dstate["player_char"]
            act2 = self.agents[agent_idx].pick_action(state, self.env.action_space, self.env.observation_space)
            orig_act = act2.copy()
            if dstate["character_locations"][act2[0]][0] == 0: act2[1] = 3 # Character in leftmost row, try to move left
            elif dstate["character_locations"][act2[0]][0] == 3: act2[1] = 2 # Character in rightmost row, try to move right
            elif dstate["character_locations"][act2[0]][1] == 0: act2[1] = 1 # Character in bottommost row, try to move down
            elif dstate["character_locations"][act2[0]][1] == 2: act2[1] = 0 # Character in topmost row, try to move up
            if dstate["character_locations"][act2[2]][0] == 0: act2[3] = 3 # Character in leftmost row, try to move left
            elif dstate["character_locations"][act2[2]][0] == 3: act2[3] = 2 # Character in rightmost row, try to move right
            elif dstate["character_locations"][act2[2]][1] == 0: act2[3] = 1 # Character in bottommost row, try to move down
            elif dstate["character_locations"][act2[2]][1] == 2: act2[3] = 0 # Character in topmost row, try to move up
            obs, reward, done, info = self.env.step(act2)
            if not np.array_equal(orig_act, act2): mod = True
            self.assertTrue(mod == False or reward == -10) # Fails unless invalid action value is caught
        # Test Valid Actions (Play remainder of game)
        while True:
            agent_idx, state = self.env.observe()
            act = self.agents[agent_idx].pick_action(state, self.env.action_space, self.env.observation_space)
            obs, reward, done, info = self.env.step(act)
            self.assertFalse(reward == -10) # Fails if any invalid actions
            if done:
                break

    # Runs after each test
    def tearDown(self):
        self.env.close()


################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    unittest.main()
