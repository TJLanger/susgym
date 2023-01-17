#!/usr/bin/env python

################################################################################
################################################################################
## Authorship
################################################################################
################################################################################
##### Project
############### susgym
##### Name
############### int_agent.py
##### Author
############### Trenton Langer
##### Creation Date
############### 20230107
##### Description
############### Intelligence based agents for use in an OpenAI gym based environment
##### Project
############### YYYYMMDD - (CONTRIBUTOR) EXAMPLE



################################################################################
################################################################################
## Imports
################################################################################
################################################################################

# Helpers
from . import agent_helpers

# Other
import constraint
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
TODO: Various intellignet agents: entropy based character moves / questions, etc...
-> Also test combos && have "intSusAgent" with best overall combo of int methods
"""

"""
description:
-> Intelligence based agent, with game knowledge to make valid action choices
-> Additionally, only chooses valid options for character identity guesses
-> Additionally, ... entropy/etc...
"""
class intSusAgent():
    def __init__(self):
        self.reward = 0
        self.num_players = None
        self.num_characters = 10

    def pick_action(self, state, act_space, obs_space):
        # Setup
        action = np.zeros(act_space.shape, dtype=np.int8)
        # State Decode
        num_players, ii, bank_gems, pg, char_locs, die_rolls, player_char, act_cards, knowledge = agent_helpers.decode_state(state, self.num_characters)
        if self.num_players is None: self.num_players = num_players
        # Action Creation
        if np.any(bank_gems == 0):
            # Character Identity Guesses
            num_opps = self.num_players-1
            for opp_idx in range(0, num_opps):
                valid_opp_chars = np.where(knowledge[opp_idx] == 1)[0]
                action[0-(num_opps-opp_idx)] = valid_opp_chars[np.random.randint(valid_opp_chars.size)]
        else:
            # Character Die Moves
            agent_helpers.randActComp_dieMove(action, die_rolls, char_locs, self.num_characters)
            # Action Card Selection
            act_card_idx = np.random.randint(0, 2)
            act_order = np.random.randint(0, 2)
            agent_helpers.randActComp_actCards(action, state, act_card_idx, act_order, self.num_characters)
        # Return
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
-> Additionally, chooses character identity guesses using constraint solver
"""
class constraintGuessSusAgent():
    def __init__(self):
        self.reward = 0
        self.num_players = None
        self.num_characters = 10

    def pick_action(self, state, act_space, obs_space):
        # Setup
        action = np.zeros(act_space.shape, dtype=np.int8)
        # State Decode
        num_players, ii, bank_gems, pg, char_locs, die_rolls, player_char, act_cards, knowledge = agent_helpers.decode_state(state, self.num_characters)
        if self.num_players is None: self.num_players = num_players
        # Action Creation
        if np.any(bank_gems == 0): # Character Identity Guesses
            # Setup
            num_opps = self.num_players-1
            opp_guess = {}
            # Iterate CSP, fix most likely guess, repeat
            while(np.any([opp_idx not in opp_guess for opp_idx in range(num_opps)])):
                # Constraint Solver Setup
                problem = constraint.Problem()
                for opp_idx in range(0, num_opps):
                    # valid_opp_chars = np.where(knowledge[opp_idx] == 1)[0]
                    valid_opp_chars = [opp_guess[opp_idx]] if opp_idx in opp_guess else np.where(knowledge[opp_idx] == 1)[0]
                    problem.addVariable(opp_idx, valid_opp_chars)
                problem.addConstraint(constraint.AllDifferentConstraint())
                # Count Identity Occurences
                solutions = problem.getSolutions()
                solution_counts = {}
                for solution in solutions:
                    for char, guess in solution.items():
                        if char not in solution_counts:
                            solution_counts[char] = {}
                        solution_counts[char][guess] = solution_counts[char].get(guess, 0) + 1
                # Guess most common solution
                # for opp_idx in range(0, num_opps):
                #     action[0-(num_opps-opp_idx)] = max(solution_counts[opp_idx], key=solution_counts[opp_idx].get)
                max_opp, max_char, max_cnt = None, None, 0
                for opp_idx in range(0, num_opps):
                    if opp_idx in opp_guess:
                        continue
                    opp_max_char = max(solution_counts[opp_idx], key=solution_counts[opp_idx].get)
                    opp_max_cnt = solution_counts[opp_idx][opp_max_char]
                    if opp_max_cnt > max_cnt:
                        max_opp = opp_idx
                        max_char = opp_max_char
                        max_cnt = opp_max_cnt
                opp_guess[max_opp] = max_char
            # Apply Guesses
            for opp_idx in range(0, num_opps):
                action[0-(num_opps-opp_idx)] = opp_guess[opp_idx]
        else:
            # Character Die Moves
            agent_helpers.randActComp_dieMove(action, die_rolls, char_locs, self.num_characters)
            # Action Card Selection
            act_card_idx = np.random.randint(0, 2)
            act_order = np.random.randint(0, 2)
            agent_helpers.randActComp_actCards(action, state, act_card_idx, act_order, self.num_characters)
        # Return
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
-> Additionally, only chooses valid options for character identity guesses
"""
class validGuessSusAgent():
    def __init__(self):
        self.reward = 0
        self.num_players = None
        self.num_characters = 10

    def pick_action(self, state, act_space, obs_space):
        # Setup
        action = np.zeros(act_space.shape, dtype=np.int8)
        # State Decode
        num_players, ii, bank_gems, pg, char_locs, die_rolls, player_char, act_cards, knowledge = agent_helpers.decode_state(state, self.num_characters)
        if self.num_players is None: self.num_players = num_players
        # Action Creation
        if np.any(bank_gems == 0):
            # Character Identity Guesses
            num_opps = self.num_players-1
            for opp_idx in range(0, num_opps):
                valid_opp_chars = np.where(knowledge[opp_idx] == 1)[0]
                action[0-(num_opps-opp_idx)] = valid_opp_chars[np.random.randint(valid_opp_chars.size)]
        else:
            # Character Die Moves
            agent_helpers.randActComp_dieMove(action, die_rolls, char_locs, self.num_characters)
            # Action Card Selection
            act_card_idx = np.random.randint(0, 2)
            act_order = np.random.randint(0, 2)
            agent_helpers.randActComp_actCards(action, state, act_card_idx, act_order, self.num_characters)
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
