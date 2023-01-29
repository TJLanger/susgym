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

# Testing
import unittest

# Helpers
from . import agent_helpers

# Other
import constraint
import numpy as np
import math

# Debug
from pprint import pprint


################################################################################
################################################################################
## Subroutines
################################################################################
################################################################################

"""
description:
-> Calculates entropy for list of values
-> E = -sum(probability*log2(probability))
parameters:
-> values: list of values
return:
-> entropy: entropy of data uniformity
"""
def entropy(*vals):
    cnts = {}
    for val in vals:
        if val not in cnts:
            cnts[val] = 0
        cnts[val] += 1
    rtn = 0
    for val, cnt in cnts.items():
        percent = cnt/len(vals)
        rtn -= percent*math.log2(percent)
    return rtn

"""
description: randomly selects indices from decoded state information
parameters:
-> decoded_state: dict of state information
-> action: numpy array to modify
return:
-> No return. Modifies action in place
"""
def randValidGuess(dstate, act):
    num_opps = dstate["num_players"]-1
    for opp_idx in range(0, num_opps):
        valid_opp_chars = np.where(dstate["knowledge"][opp_idx] == 1)[0]
        act[0-(num_opps-opp_idx)] = valid_opp_chars[np.random.randint(valid_opp_chars.size)]



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
class intSusAgent(agent_helpers.susAgent):
    def _act_charGuess(self, decoded_state, action, act_space, obs_space):
        randValidGuess(decoded_state, action)

    def _act_actCards(self, decoded_state, action, act_space, obs_space):
        # Random Action Selection # TODO: update to best method from other int agents
        super()._act_actCards(self, decoded_state, action, act_space, obs_space)
"""
description:
-> Intelligence based agent, with game knowledge to make valid action choices
-> Stochastic, but valid, character identity guesses
-> Randomly selects card/actions, but updates opponent in character ask cards to maxmimize information gain
"""
class entropyAskSusAgent(agent_helpers.susAgent):
    def _act_charGuess(self, decoded_state, action, act_space, obs_space):
        randValidGuess(decoded_state, action)

    def _act_actCards(self, decoded_state, action, act_space, obs_space):
        # Random Action Selection
        super()._act_actCards(decoded_state, action, act_space, obs_space)
        # Update LineOfSightQuestion target, if applicable
        if np.any(decoded_state["action_cards"][action[4]][6:16] == 1):
            # Setup
            action[5] = 0 # force action order to question being last (after potential trapdoor), so dstate["charcter_locations"] is accurate from helper updates
            qchar = np.where(decoded_state["action_cards"][action[4]][6:16] == 1)[0][0] # Character on question card
            # Evaluate Opponents
            entropies = []
            for opp_kb in decoded_state["knowledge"]:
                line_of_sight = []
                for char_idx in range(len(opp_kb)):
                    if opp_kb[char_idx] == 1: # Only include characters opponent could be
                        if decoded_state["character_locations"][qchar][0] == decoded_state["character_locations"][char_idx][0] or decoded_state["character_locations"][qchar][1] == decoded_state["character_locations"][char_idx][1]:
                            line_of_sight.append(True)
                        else:
                            line_of_sight.append(False)
                entropies.append(entropy(*line_of_sight))
            # Update Target
            max_e = np.argmax(np.array(entropies))
            action[7] = 6*(max_e/(decoded_state["num_players"]-1)) # Question card always uses target 2

"""
description:
-> Stochastic agent, with game knowledge to make random (but valid) action choices
-> Additionally, chooses character identity guesses using constraint solver
"""
class constraintGuessSusAgent(agent_helpers.susAgent):
    def _act_charGuess(self, decoded_state, action, act_space, obs_space):
        # Setup
        num_opps = decoded_state["num_players"]-1
        opp_guess = {}
        # Iterate CSP, fix most likely guess, repeat
        while(np.any([opp_idx not in opp_guess for opp_idx in range(num_opps)])):
            # Constraint Solver Setup
            problem = constraint.Problem()
            for opp_idx in range(0, num_opps):
                # valid_opp_chars = np.where(knowledge[opp_idx] == 1)[0]
                valid_opp_chars = [opp_guess[opp_idx]] if opp_idx in opp_guess else np.where(decoded_state["knowledge"][opp_idx] == 1)[0]
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

"""
description:
-> Stochastic agent, with game knowledge to make random (but valid) action choices
-> Additionally, only chooses valid options for character identity guesses
"""
class smartGuessSusAgent(agent_helpers.susAgent):
    def _act_charGuess(self, decoded_state, action, act_space, obs_space):
        # Setup
        num_opps = decoded_state["num_players"]-1
        opp_guess = {}
        # Iterate, fix most likely guess, repeat
        while(np.any([opp_idx not in opp_guess for opp_idx in range(num_opps)])):
            min_opp, min = None, None # Opponent with fewest possibilities
            for opp_idx in range(0, num_opps):
                if opp_idx in opp_guess:
                    continue
                opp_sum = np.sum(decoded_state["knowledge"][opp_idx])
                if min_opp is None or opp_sum < min:
                    min_opp = opp_idx
                    min = opp_sum
            # Generate Guess
            valid_opp_chars = np.where(decoded_state["knowledge"][opp_idx] == 1)[0]
            if len(valid_opp_chars) == 0: valid_opp_chars = np.array([x for x in range(10)]) # Refill if guesses eliminate all possibilities
            opp_guess[min_opp] = valid_opp_chars[np.random.randint(valid_opp_chars.size)]
            for opp_idx in range(0, num_opps):
                if opp_idx == min_opp: continue
                decoded_state["knowledge"][opp_idx][opp_guess[min_opp]] = 0 # Eliminate guess from other players
        # Apply Guesses
        for opp_idx in range(0, num_opps):
            action[0-(num_opps-opp_idx)] = opp_guess[opp_idx]

"""
description:
-> Stochastic agent, with game knowledge to make random (but valid) action choices
-> Additionally, only chooses valid options for character identity guesses
"""
class validGuessSusAgent(agent_helpers.susAgent):
    def _act_charGuess(self, decoded_state, action, act_space, obs_space):
        randValidGuess(decoded_state, action)


################################################################################
################################################################################
## Tests
################################################################################
################################################################################

"""
description:
-> Entropy Calculation Testing
"""
class TestEntropy(unittest.TestCase):
    def test_basics(self):
        uniform_vals = (1 for _ in range(5))
        e_uniform = entropy(*uniform_vals)
        self.assertEqual(e_uniform, 0, "Entropy of uniform set not equal to 0")
        split_vals = (int(x/5) for x in range(10))
        e_split = entropy(*split_vals)
        self.assertEqual(e_split, 1, "Entropy of 50/50 split set not equal to 1")



################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("File meant for use as module, executing unit tests")
    unittest.main()
