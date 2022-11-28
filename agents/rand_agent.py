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
            curr_loc = char_locs[2*roll:2*roll+2]
            action[act_idx+1:act_idx+3] = curr_loc
            while True:
                x_or_y = np.random.randint(0, 2)
                plus_or_minus = 1 if np.random.randint(0, 2) == 1 else -1
                new_val = action[act_idx+1+x_or_y] + plus_or_minus
                if x_or_y == 0 and new_val >= 0 and new_val < 4:
                    break
                elif x_or_y == 1 and new_val >= 0 and new_val < 3:
                    break
            action[act_idx+1+x_or_y] += plus_or_minus
            char_locs[2*roll:2*roll+2] = action[act_idx+1:act_idx+3]
            act_idx += 3
        act_card_idx = np.random.randint(0, 2)
        act_order = np.random.randint(0, 2)
        card_act_idxs = np.where(act_cards[act_card_idx] == 1)[0]
        for act_card_action_idx in range(len(card_act_idxs)):
            # Setup
            act_card_action = card_act_idxs[act_card_action_idx] if act_order == 0 else card_act_idxs[len(card_act_idxs)-1-act_card_action_idx]
            # Set Action
            action[act_idx] = act_card_idx
            if act_card_action == 0: # Trapdoor action
                action[act_idx+7] = 1 + np.random.randint(0, self.num_characters)
                action[act_idx+8] = np.random.randint(0, 4)
                action[act_idx+9] = np.random.randint(0, 3)
            elif act_card_action <= 3: # Lucky Lift
                action[act_idx+act_card_action] = 1
            elif act_card_action == 4: # Room Gem Take
                action[act_idx+4] = 1
                valid_room_gems = np.where(room_gems == 1)[0]
                action[act_idx+1+valid_room_gems[np.random.randint(0,len(valid_room_gems))]] = 1
            elif act_card_action == 5: # View invite deck
                action[act_idx+5] = 1
            else: # Character ask (line of sight)
                char_to_ask = act_card_action - 6
                action[act_idx+6] = np.random.randint(1, self.num_players) # 0 indicates self (no ask), all others are opponent offset from current player
            # Increment act idx
            act_idx += 10
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
