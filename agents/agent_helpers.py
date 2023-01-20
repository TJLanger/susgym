#!/usr/bin/env python

################################################################################
################################################################################
## Authorship
################################################################################
################################################################################
##### Project
############### susgym
##### Name
############### agent_helpers.py
##### Author
############### Trenton Langer
##### Creation Date
############### 20230107
##### Description
############### Common functionality for OpenAI Gym SuspicionEnv agents
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

"""
description:
-> checks if an action is valid for the given game state and settings
parameters:
-> state: numpy array containing state values
-> num_chars: int representing number of characters in the game
return:
-> invite_idx: index of the deck of remaining invitation cards
-> bank_gems: array of remaining gems for each color (Red, Green, Yellow)
-> player_gems: 2D array, with accumulated gem counts for each player (R,G,Y)
-> character_locations: 2D array, with x/y location of each character
-> die_rolls: Array containing character number rolled for each die (0->#C, #C = ? roll)
-> player_char: player character assignment
-> action_cards: 2D array of action cards
-> knowledge: 2D array of boolean indicator for each opponents possible character assignments
"""
def decode_state(state, num_characters=10):
    # Setup
    state_idx = 0 # Increment variable, to step through state array
    player_gems = []
    character_locations = []
    action_cards = []
    knowledge = []

    # Determine Player Count
    state_len = len(state)
    state_len -= (1 + 3 + (2*num_characters) + 2 + 1 + (2*16)) # Subtract non player specific
    num_players = int((state_len + num_characters) / (3 + num_characters))

    # State Decode
    invite_idx = state[state_idx]
    state_idx += 1

    bank_gems = state[state_idx:state_idx+3]
    state_idx += 3

    for player_idx in range(num_players):
        player_gems.append(state[state_idx:state_idx+3])
        state_idx += 3 # increment past gem counts for all players

    for character_idx in range(num_characters):
        character_locations.append(state[state_idx:state_idx+2])
        state_idx += 2

    die_rolls = state[state_idx:state_idx+2]
    state_idx += 2

    player_char = state[state_idx]
    state_idx += 1

    for card in range(2):
        action_cards.append(state[state_idx:state_idx+16])
        state_idx += 16

    for opponent_idx in range(num_players-1):
        knowledge.append(state[state_idx:state_idx+num_characters])
        state_idx += num_characters

    # Return
    return num_players, invite_idx, bank_gems, player_gems, character_locations, die_rolls, player_char, action_cards, knowledge

"""
description:
-> returns 2d array of form arr[roomx][roomy] = [gems boolean]
parameters:
-> No Params
return:
-> room_gems: array of boolean indicators
"""
def get_room_gems():
    # Shape of X, Y, [Red, Green, Yellow]
    board = np.zeros((4,3,3), dtype=np.int8)
    board[0][0][0:3] = [1, 0, 1]
    board[1][0][0:3] = [0, 1, 0]
    board[2][0][0:3] = [1, 1, 0]
    board[3][0][0:3] = [0, 1, 1]
    board[0][1][0:3] = [0, 1, 1]
    board[1][1][0:3] = [1, 1, 0]
    board[2][1][0:3] = [1, 0, 1]
    board[3][1][0:3] = [0, 0, 1]
    board[0][2][0:3] = [1, 0, 0]
    board[1][2][0:3] = [1, 0, 1]
    board[2][2][0:3] = [0, 1, 1]
    board[3][2][0:3] = [1, 1, 0]
    return board

"""
description:
-> checks if an action is valid for the given game state and settings
parameters:
-> action: numpy array containing action values
-> state: numpy array containing state values
-> num_players: int representing number of players in the game
-> num_chars: int representing number of characters in the game. Default=10
-> board_width: int representing number of rooms board is wide. Default=4
-> board_height: int representing number of rooms board is tall. Default=3
return:
-> isValid, boolean indicating if action was valid or not
"""
def validate_action(action, state, num_characters=10, board_width=4, board_height=3):
    return False # TODO: Update to match new action space
    # setup
    act_idx = 0
    # Get State info
    num_players, invite_idx, bank_gems, player_gems, char_locs, die_rolls, player_char, act_cards, knowledge = decode_state(state, num_characters)
    room_gems = get_room_gems()[char_locs[player_char][0]][char_locs[[player_char]][1]]
    # Check if normal gameplay or endgame (guessing identities)
    if np.all(bank_gems > 0):
        # Check Die Moves
        for die in range(2):
            # Setup
            die_move = action[act_idx:act_idx+3] # char, new_x, new_y
            act_idx += 3
            # Check on board
            if die_move[1] < 0 or die_move[1] >= board_width or die_move[2] < 0 or die_move[2] >= board_height:
                return False
            # Check Move
            if die_move[0] in die_rolls:
                die_rolls = np.delete(die_rolls, np.where(die_rolls == die_move[0])[0][0])
                if die_move[1] == char_locs[die_move[0]][0] and ((die_move[2] == char_locs[die_move[0]][1] - 1) or (die_move[2] == char_locs[die_move[0]][1] + 1)):
                    char_locs[die_move[0]] = die_move[1:3] # update char_loc for "?" followed by regular roll
                elif die_move[2] == char_locs[die_move[0]][1] and ((die_move[1] == char_locs[die_move[0]][0] - 1) or (die_move[1] == char_locs[die_move[0]][0] + 1)):
                    char_locs[die_move[0]] = die_move[1:3] # update char_loc for "?" followed by regular roll
                else:
                    return False
            elif num_characters in die_rolls:
                die_rolls = np.delete(die_rolls, np.where(die_rolls == num_characters)[0][0])
                if die_move[1] == char_locs[die_move[0]][0] and ((die_move[2] == char_locs[die_move[0]][1] - 1) or (die_move[2] == char_locs[die_move[0]][1] + 1)):
                    char_locs[die_move[0]] = die_move[1:3] # update char_loc for "?" followed by regular roll
                elif die_move[2] == char_locs[die_move[0]][1] and ((die_move[1] == char_locs[die_move[0]][0] - 1) or (die_move[1] == char_locs[die_move[0]][0] + 1)):
                    char_locs[die_move[0]] = die_move[1:3] # update char_loc for "?" followed by regular roll
                else:
                    return False
            else:
                return False # attempted to move invalid character
        # Check Action Card Actions
        for idx in range(2):
            # Setup
            act_card_action = action[act_idx:act_idx+10]
            ###print("Act Card Action: %s" % act_card_action)
            act_idx += 10
            # Validate Gem take
            if np.any(act_card_action[1:5]):
                if act_card_action[4] == 1 and np.sum(act_card_action[1:4]) == 0:
                    return False # Room flag set, but no gems marked for taking
                gem_idx = np.where(act_card_action[1:4] == 1)[0][0]
                if act_card_action[4] > 0 and not room_gems[gem_idx] == 1: # Taking from room
                    ###print("Bad Room Take")
                    return False
                elif act_card_action[4] == 0 and not act_cards[act_card_action[0]][1+gem_idx] == 1: # lucky lift
                    ###print("Bad Lucky Take (%s)" % gem_idx)
                    return False
            # Validate Invite Deck View
            if act_card_action[5] == 1 and not act_cards[act_card_action[0]][5] == 1:
                ###print("Bad Invite")
                return False
            # Validate Question Player
            if act_card_action[6] != 0 and not np.any(act_cards[act_card_action[0]][-num_characters:] == 1):
                ###print("Bad Question")
                return False
            # Validate trapdoor
            if act_card_action[7] > 0 and not act_cards[act_card_action[0]][0] == 1:
                ###print("Bad Trapdoor")
                return False
    else: # Only need identity guesses
        pass # Any identity guess that meets the action space requirements is valid
    # Return Valid
    return True

"""
description:
-> updates given action array with stochastic character identity guesses
parameters:
-> action: numpy array containing action values
-> num_players: int representing number of players in the game
-> num_chars: int representing number of characters in the game. Default=10
return:
-> No Return
"""
def randActComp_charGuess(action, num_players, num_characters=10):
    for opp_idx in range(1, num_players):
        action[0-opp_idx] = np.random.randint(num_characters)

"""
description:
-> updates given action array with stochastic character die moves
parameters:
-> action: numpy array containing action values
-> die_rolls: Array containing character number rolled for each die (0->#C, #C = ? roll)
-> character_locations: 2D array, with x/y location of each character
-> num_chars: int representing number of characters in the game. Default=10
-> board_width: int representing number of rooms board is wide. Default=4
-> board_height: int representing number of rooms board is tall. Default=3
return:
-> No Return
"""
def randActComp_dieMove(action, die_rolls, character_locations, num_characters=10, board_width=4, board_height=3):
    # Setup
    act_idx = 0 # Start updating at beginning of action array
    die_rolls_charOnly = list(map(lambda x: np.random.randint(0, num_characters) if x == num_characters else x, list(die_rolls)))
    curr_locs = np.copy(character_locations)
    # Apply Roll
    for roll in die_rolls_charOnly:
        action[act_idx] = roll
        while True:
            x_or_y = np.random.randint(0, 2)
            plus_or_minus = 1 if np.random.randint(0, 2) == 1 else -1
            new_val = curr_locs[roll][x_or_y] + plus_or_minus
            if x_or_y == 0 and new_val >= 0 and new_val < board_width:
                break
            elif x_or_y == 1 and new_val >= 0 and new_val < board_height:
                break
        curr_locs[roll][x_or_y] += plus_or_minus
        action[act_idx] = roll
        x_offset = 0 if plus_or_minus > 0 else 1
        action[act_idx+1] =  2*(1-x_or_y) + x_offset #(up=0, down=1, right=2, left=3)
        act_idx += 2

"""
description:
-> updates given action array with stochastic action card choices
parameters:
-> action: numpy array containing action values
-> num_players: int representing number of players in the game
-> state: state array
-> act_card_idx: int, selecting which action card in hand to apply. Default=random
-> act_order: int, selecting top/bottom of action card to apply first. Default=random
-> num_chars: int representing number of characters in the game. Default=10
-> board_width: int representing number of rooms board is wide. Default=4
-> board_height: int representing number of rooms board is tall. Default=3
return:
-> No return
"""
def randActComp_actCards(action, state, act_card_idx=None, act_order=None, num_characters=10, board_width=4, board_height=3):
    # Setup
    act_idx = 0 # Read && apply die rolls, before moving on and updating for action cards
    # Decode State
    num_players, invite_idx, bank_gems, player_gems, char_locs, die_rolls, player_char, act_cards, knowledge = decode_state(state, num_characters)
    room_gems = get_room_gems()[char_locs[player_char][0]][char_locs[player_char][1]]
    # Update for Die Rolls
    char_loc_start = 1 + 3*(num_players+1)
    for roll_idx in range(2):
        # Setup
        die_move = action[act_idx:act_idx+2] # char, direction (up, down, right, left)
        move_dir = ([1,0] if die_move[1] % 2 == 0 else [-1,0]) if die_move[1] > 1 else ([0,1] if die_move[1] % 2 == 0 else [0,-1])
        move_char = die_move[0] if die_rolls[roll_idx] == num_characters else die_rolls[roll_idx]
        # Update Char Locs
        char_locs[move_char] += move_dir # WARNING - Char_locs and state linked by reference - updates both
        # Update Iteration Idx
        act_idx += 2
    room_gems = get_room_gems()[char_locs[player_char][0]][char_locs[player_char][1]]
    # Action Card Selection
    if act_card_idx is None: act_card_idx = np.random.randint(0, 2) # random select one of two action cards
    if act_order is None: act_order = np.random.randint(0, 2) # random select top or bottom action to apply first
    card_act_idxs = np.where(act_cards[act_card_idx] == 1)[0]
    if act_order == 1:
        card_act_idxs = np.flip(card_act_idxs)
    action[act_idx] = act_card_idx
    act_idx += 1
    action[act_idx] = act_order
    act_idx += 1
    # Apply Action Card
    for act_card_action_idx in range(len(card_act_idxs)):
        # Setup
        act_card_action = card_act_idxs[act_card_action_idx]
        target_max = 120 if act_order == 0 else 6
        # Set Action
        if act_card_action == 0: # Trapdoor action
            # Random Select Char/Loc
            td_char = np.random.randint(0, num_characters)
            td_x = np.random.randint(0, board_width)
            td_y = np.random.randint(0, board_height)
            # Update Action
            action[act_idx+act_order] += 12*td_char
            action[act_idx+act_order] += 3*td_x
            action[act_idx+act_order] += td_y
            # Update State (If TD occurs prior to gem take)
            char_locs[td_char] = [td_x, td_y]
        elif act_card_action <= 3: # Lucky Lift
            pass # Target irrelevant for lucky lift -> action determined completely by action card
        elif act_card_action == 4: # Room Gem Take
            room_gems = get_room_gems()[char_locs[player_char][0]][char_locs[player_char][1]]
            valid_room_gems = np.where(room_gems == 1)[0]
            take_gem = valid_room_gems[np.random.randint(0,len(valid_room_gems))]
            # print("Act Helper, Take Gem: %s, from %s" % (take_gem, valid_room_gems))
            action[act_idx+act_order] = int((target_max/3)*(take_gem+0.5)) # "-1" to shift idx'ing to start at 0, '+0.5' to put output values in middle of range
        elif act_card_action == 5: # View invite deck
            pass # Target irrelevant to invite deck view, no options but top card to view
        else: # Character ask (line of sight)
            player_offset = np.random.randint(0, num_players-1) # Offset target value off by 1 (0 offset target == next player, 1 == next next player, ...)
            action[act_idx+act_order] = target_max*(player_offset/(num_players-1))
        # Increment act idx
        act_order = 1 - act_order # Update order to pick other target for next action

################################################################################
################################################################################
## Classes
################################################################################
################################################################################

"""
description:
-> Replay Buffer for storing state/action/reward/state' tuples
"""
class ReplayBuffer:
    def __init__(self, capacity=100000, batch_size=64):
        # Save Params
        self.capacity = capacity # max number of observations to save
        self.batch_size = batch_size # number of observations to return when queried
        # Internal Vars
        self.counter = 0 # count insertions, to determine index to write/overwrite
        self.buffer_obs = None # class variable to store states in
        self.buffer_act = None # class variable to store actions in
        self.buffer_rwd = np.zeros((self.capacity, 1), dtype=np.float32) # class variable to store rewards in, not dependent on act/obs space
        self.buffer_nob = None # class variable to store next states in

    def __len__(self):
        return min(self.counter, self.capacity)

    def store(self, obs_tuple):
        # Setup
        index = self.counter % self.capacity # loops when count >= capacity
        if self.buffer_obs is None: # Create act/obs space dependent buffers on first addition to replaybuffer
            self.buffer_obs = np.zeros((self.capacity, obs_tuple[0].shape[0]), dtype=np.float32)
            self.buffer_act = np.zeros((self.capacity, obs_tuple[1].shape[0]), dtype=np.float32)
            self.buffer_nob = np.zeros((self.capacity, obs_tuple[3].shape[0]), dtype=np.float32)
        # Store
        self.buffer_obs[index] = obs_tuple[0]
        self.buffer_act[index] = obs_tuple[1]
        self.buffer_rwd[index] = obs_tuple[2]
        self.buffer_nob[index] = obs_tuple[3]
        # Update
        self.counter += 1

    def sample(self):
        # Setup
        sample_range = min(self.counter, self.capacity)
        # Random select indices
        idxs = np.random.choice(sample_range, self.batch_size)
        # Return
        return self.buffer_obs[idxs].copy(), self.buffer_act[idxs].copy(), self.buffer_rwd[idxs].copy(), self.buffer_nob[idxs].copy()



################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("No executable code, meant for use as module only")
