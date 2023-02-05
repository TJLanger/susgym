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

# Testing
import unittest

# OpenAI gym
import gym

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
-> parses state array into understandable/labeled pieces
parameters:
-> state: numpy array containing state values
-> num_chars: int representing number of characters in the game
return:
---> decoded state dict
---> invite_idx: index of the deck of remaining invitation cards
---> bank_gems: array of remaining gems for each color (Red, Green, Yellow)
---> player_gems: 2D array, with accumulated gem counts for each player (R,G,Y)
---> character_locations: 2D array, with x/y location of each character
---> die_rolls: Array containing character number rolled for each die (0->#C, #C = ? roll)
---> player_char: player character assignment
---> action_cards: 2D array of action cards
---> knowledge: 2D array of boolean indicator for each opponents possible character assignments
"""
def decode_state(state, num_characters=10):
    # Setup
    state_idx = 0 # Increment variable, to step through state array
    decoded_state = {
        "player_gems":[],
        "character_locations":[],
        "action_cards":[],
        "knowledge":[],
    }
    if not isinstance(state, np.ndarray):
        state = np.array(state)

    # Determine Player Count
    state_len = len(state)
    state_len -= (1 + 3 + (2*num_characters) + 2 + 1 + (2*16)) # Subtract non player specific
    decoded_state["num_players"] = int((state_len + num_characters) / (3 + num_characters))

    # State Decode
    decoded_state["invite_idx"] = state[state_idx]
    state_idx += 1

    decoded_state["bank_gems"] = state[state_idx:state_idx+3]
    state_idx += 3

    for player_idx in range(decoded_state["num_players"]):
        decoded_state["player_gems"].append(state[state_idx:state_idx+3])
        state_idx += 3 # increment past gem counts for all players

    for character_idx in range(num_characters):
        decoded_state["character_locations"].append(state[state_idx:state_idx+2])
        state_idx += 2

    decoded_state["die_rolls"] = state[state_idx:state_idx+2]
    state_idx += 2

    decoded_state["player_char"] = state[state_idx]
    state_idx += 1

    for card in range(2):
        decoded_state["action_cards"].append(state[state_idx:state_idx+16])
        state_idx += 16

    for opponent_idx in range(decoded_state["num_players"]-1):
        decoded_state["knowledge"].append(state[state_idx:state_idx+num_characters])
        state_idx += num_characters

    # Return
    return decoded_state

"""
description:
-> returns 2D numpy array with all possible actions for act_space, prefixed by valid boolean flag
-> doesnt roll out all combos of opponent guesses, too many possiblities, and shouldnt be relevant to network
-> uses game knowledge to not roll out duplicate choices (multiple target values for same gem/opponent)
parameters:
-> act_space: Gym environment action space, type MutliDiscrete
-> state: numpy array representing state
-> num_player:
return:
-> actions: 2D numpy array of form [[0/1,action],[0/1,action],...]
"""
def get_valid_actions(act_space: gym.spaces.MultiDiscrete, state: np.ndarray, num_players: int) -> np.ndarray:
    # Setup
    act_size, state_size = len(act_space.nvec), len(state)
    char_loc_start = 1 + 3*(num_players+1)
    die_start = char_loc_start + 2*10
    card_start = die_start + 3

    no = num_players - 1 # num_opponent
    card_act_sizes = [120, 1,1,1, 3, 1, no,no,no,no,no,no,no,no,no,no]# trapdoor getRedGem getGreen getYellow getRoom viewDeck 10xaskCharacter (Alphabetical)

    # State Checks
    die1_chars = 10 if state[die_start] == 10 else 1 # number of chars to rollout for die1
    die2_chars = 10 if state[die_start+1] == 10 else 1
    card_acts = np.zeros((2,2), dtype=np.int8)
    card_acts[0] = np.where(state[card_start:card_start+16] == 1)[0]
    card_acts[1] = np.where(state[card_start+16:card_start+16+16] == 1)[0]

    # Card Rollouts
    rtn = []
    for card_idx in range(2):
        # Setup
        rollout = [die1_chars,4,die2_chars,4,1,2,card_act_sizes[card_acts[card_idx][0]],card_act_sizes[card_acts[card_idx][1]]]
        # Calc Num Actions
        act_len = 1
        for size in rollout:
            act_len *= size
        # Create Array
        actions = np.zeros((act_len,1+act_size+state_size), dtype=np.float64)
        actions[:,0] = 1 # Default all actions to valid
        write_idx = 1 # Skip boolean valid/invalid flag preceding action
        write_size = act_len
        for size in rollout:
            # Create Overwrite
            ovr = np.tile(np.repeat(np.array([x for x in range(size)], dtype=np.int16), int(write_size/size)), int(act_len/write_size))
            actions[:,write_idx] = ovr.T
            # Updates
            write_idx += 1
            write_size /= size
        # State Specific Values
        if die1_chars == 1: actions[:,1+0] = state[die_start]
        if die2_chars == 1: actions[:,1+2] = state[die_start+1]
        actions[:,1+4] = card_idx
        actions[:,1+6] += 0.5
        actions[:,1+7] += 0.5
        actions[:,1+6] *= int(120/rollout[6])
        actions[:,1+7] *= int(6/rollout[7])
        actions = actions.astype(np.int16)
        # Extend with State Copies
        actions[:,1+act_size:] = np.repeat(state[np.newaxis,:], act_len, axis=0)
        ### Validate Character Direction, Die Roll 1
        valid = np.where(np.any([
            ((actions[:,2] == 0) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,1]+1] < 2)), # Up
            ((actions[:,2] == 1) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,1]+1] > 0)),  # Down
            ((actions[:,2] == 2) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,1]] < 3)), # Right
            ((actions[:,2] == 3) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,1]] > 0)) # Left
        ], axis=0), 1, 0)
        actions[:,0] = np.minimum(actions[:,0],valid.T)
        ### State Update (applied regardless of if valid)
        for update in ((0,1,1),(1,1,-1),(2,0,1),(3,0,-1),): # up:(dir=0,x/y=1,delta=1), down:(dir=1,x/y=1,delta=-1)
            update_idx = np.where(actions[:,2] == update[0])
            actions[update_idx,1+act_size+char_loc_start+2*actions[update_idx,1]+update[1]] += update[2]
        ### Character Direction, Die Roll 2
        valid = np.where(np.any([
            ((actions[:,4] == 0) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,3]+1] < 2)), # Up
            ((actions[:,4] == 1) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,3]+1] > 0)), # Down
            ((actions[:,4] == 2) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,3]] < 3)), # Right
            ((actions[:,4] == 3) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,3]] > 0)) # Left
        ], axis=0), 1, 0)
        actions[:,0] = np.minimum(actions[:,0],valid.T)
        ### State Update (applied regardless of if valid)
        for update in ((0,1,1),(1,1,-1),(2,0,1),(3,0,-1),): # up:(dir=0,x/y=1,delta=1), down:(dir=1,x/y=1,delta=-1)
            update_idx = np.where(actions[:,4] == update[0])
            actions[update_idx,1+act_size+char_loc_start+2*actions[update_idx,3]+update[1]] += update[2]
        ### Trapdoor State Update iff act_order==0 (else TD after gem takes - irrelevant/all valid)
        td_updates = np.where((actions[:,1+act_size+card_start+16*card_idx] == 1) & (actions[:,1+5] == 0))
        char_updates = actions[td_updates,1+6] // 12 # floor divide by number of rooms = target character
        td_rooms = actions[td_updates,1+6] % 12 # modulus by number of rooms to get room number
        actions[td_updates,1+act_size+char_loc_start+2*char_updates] = td_rooms // 3 # new room x is floor divide of room number and board height
        actions[td_updates,1+act_size+char_loc_start+2*char_updates+1] = td_rooms % 3 # new room y is modulus of room number and board height
        ### Room Gem takes
        rooms = (
            (0,0,[0,2]),
            (1,0,[1]),
            (2,0,[0,1]),
            (3,0,[1,2]),
            (0,1,[1,2]),
            (1,1,[0,1]),
            (2,1,[0,2]),
            (3,1,[2]),
            (0,2,[0]),
            (1,2,[0,2]),
            (2,2,[1,2]),
            (3,2,[0,1]),
        )
        for room in rooms: # tuple of (room_x, room_y, valid_gem)
            room_matches = np.where((actions[:,1+act_size+card_start+16*card_idx+4] == 1) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,1+act_size+die_start+2]] == room[0]) & (actions[np.arange(act_len)[:],1+act_size+char_loc_start+2*actions[:,1+act_size+die_start+2]+1] == room[1]))[0] # using room card, and in room
            if len(room_matches) == 0: continue
            valid_rooms = np.where(np.any([
                ((np.sum(actions[room_matches,1+act_size+card_start+16*card_idx:1+act_size+card_start+16*card_idx+4], axis=1) == 0) & (np.isin(3*actions[room_matches,1+6] // 120, room[2]))), # Room gem first
                ((np.sum(actions[room_matches,1+act_size+card_start+16*card_idx:1+act_size+card_start+16*card_idx+4], axis=1) > 0) & (np.isin(3*actions[room_matches,1+7] // 6, room[2]))), # Room gem second
            ], axis=0), 1, 0)
            actions[room_matches,0] = np.minimum(actions[room_matches,0],valid_rooms.T)
        # Save
        rtn.append(actions[np.where(actions[:,0]==1)[0],1:1+act_size]) # Dropping prefix valid/invalid column on return
    # Return
    return np.concatenate(rtn, axis=0)

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
def randActComp_actCards(state, action, act_card_idx=None, act_order=None, num_characters=10, board_width=4, board_height=3):
    # Setup
    act_idx = 0 # Read && apply die rolls, before moving on and updating for action cards
    # Decode State
    dstate = state if isinstance(state, dict) else decode_state(state, num_characters)
    # Update for Die Rolls
    char_loc_start = 1 + 3*(dstate["num_players"]+1)
    for roll_idx in range(2):
        # Setup
        die_move = action[act_idx:act_idx+2] # char, direction (up, down, right, left)
        move_dir = ([1,0] if die_move[1] % 2 == 0 else [-1,0]) if die_move[1] > 1 else ([0,1] if die_move[1] % 2 == 0 else [0,-1])
        move_char = die_move[0] if dstate["die_rolls"][roll_idx] == num_characters else dstate["die_rolls"][roll_idx]
        # Update Char Locs
        dstate["character_locations"][move_char] += move_dir # WARNING - Char_locs and state linked by reference - updates both
        # Update Iteration Idx
        act_idx += 2
    room_gems = get_room_gems()[dstate["character_locations"][dstate["player_char"]][0]][dstate["character_locations"][dstate["player_char"]][1]]
    # Action Card Selection
    if act_card_idx is None: act_card_idx = np.random.randint(0, 2) # random select one of two action cards
    if act_order is None: act_order = np.random.randint(0, 2) # random select top or bottom action to apply first
    card_act_idxs = np.where(dstate["action_cards"][act_card_idx] == 1)[0]
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
            dstate["character_locations"][td_char] = [td_x, td_y]
        elif act_card_action <= 3: # Lucky Lift
            pass # Target irrelevant for lucky lift -> action determined completely by action card
        elif act_card_action == 4: # Room Gem Take
            room_gems = get_room_gems()[dstate["character_locations"][dstate["player_char"]][0]][dstate["character_locations"][dstate["player_char"]][1]]
            valid_room_gems = np.where(room_gems == 1)[0]
            take_gem = valid_room_gems[np.random.randint(0,len(valid_room_gems))]
            # print("Act Helper, Take Gem: %s, from %s" % (take_gem, valid_room_gems))
            action[act_idx+act_order] = int((target_max/3)*(take_gem+0.5)) # "-1" to shift idx'ing to start at 0, '+0.5' to put output values in middle of range
        elif act_card_action == 5: # View invite deck
            pass # Target irrelevant to invite deck view, no options but top card to view
        else: # Character ask (line of sight)
            player_offset = np.random.randint(0, dstate["num_players"]-1) # Offset target value off by 1 (0 offset target == next player, 1 == next next player, ...)
            action[act_idx+act_order] = target_max*(player_offset/(dstate["num_players"]-1))
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

    def biased_sample(self, bias=None): # Bias=2 => half batch is recent data
        # Setup
        idxs = np.zeros(self.batch_size, dtype=np.uint32) # Int needs to be able to cover entire buffer range of idxs
        sample_range = min(self.counter, self.capacity)
        partial_batch_size = 0 if bias is None else int(self.batch_size/bias) # default to no bias (0 size for partial batch)
        # Select most recent half batch
        curr_idx = self.counter % self.capacity # loops when count >= capacity
        if curr_idx >= partial_batch_size:
            idxs[0:partial_batch_size] = np.arange(curr_idx-partial_batch_size, curr_idx)
        else:
            idxs[0:curr_idx] = np.arange(0, curr_idx)
            idxs[curr_idx:partial_batch_size] = np.arange(self.capacity-(partial_batch_size-curr_idx), self.capacity)
        # Random select indices
        idxs[partial_batch_size:self.batch_size] = np.random.choice(sample_range, self.batch_size-partial_batch_size)
        # Return
        return self.buffer_obs[idxs].copy(), self.buffer_act[idxs].copy(), self.buffer_rwd[idxs].copy(), self.buffer_nob[idxs].copy()

"""
description:
-> Stochastic agent, with game knowledge to make random (but valid) action choices
-> Provides template for othher agents to follow, for overwriting portions of decisions
"""
class susAgent():
    def __init__(self):
        self.reward = 0
        self.num_characters = 10

    def update(self, next_state, reward, done, info):
        self.reward += reward

    def getReward(self):
        return self.reward

    def reset(self, reward=0):
        self.reward = reward

    def pick_action(self, state, act_space, obs_space):
        # Setup
        action = np.zeros(act_space.shape, dtype=np.int8)
        # State Decode
        dstate = decode_state(state, self.num_characters)
        # Action Creation
        if np.any(dstate["bank_gems"] == 0):
            # Character Identity Guesses
            self._act_charGuess(dstate, action, act_space, obs_space)
        else:
            # Character Die Moves
            self._act_dieMove(dstate, action, act_space, obs_space)
            # Action Card Selection
            self._act_actCards(dstate, action, act_space, obs_space)
        # Return
        return action

    def _act_charGuess(self, decoded_state, action, act_space, obs_space):
        randActComp_charGuess(action, decoded_state["num_players"], self.num_characters)

    def _act_dieMove(self, decoded_state, action, act_space, obs_space):
        randActComp_dieMove(action, decoded_state["die_rolls"], decoded_state["character_locations"], self.num_characters)

    def _act_actCards(self, decoded_state, action, act_space, obs_space):
        act_card_idx = np.random.randint(0, 2)
        act_order = np.random.randint(0, 2)
        randActComp_actCards(decoded_state, action, act_card_idx, act_order, self.num_characters)

    def close(self): # Agent cleanup
        pass



################################################################################
################################################################################
## Tests
################################################################################
################################################################################

"""
description:
-> Action Rollout (of valid) Testing
"""
class TestActRollout(unittest.TestCase):
    pass # TODO: based on state(s), manually calc possibilities, multiply, and verify lengths match up

"""
description:
-> ReplayBuffer Testing
"""
class TestReplayBuffer(unittest.TestCase):    # Runs prior to each test
    def setUp(self):
        self.cap = 1000 # Buffer capacity
        self.bs = 8 # Buffer batch size
        self.rb = ReplayBuffer(capacity=self.cap, batch_size=self.bs)

    # Test store method
    def test_store(self):
        self.rb.store((np.zeros(1),np.zeros(1),0,np.zeros(1))) # Pass test unless error

    # Test overridden len()
    def test_len(self):
        self.assertEqual(len(self.rb), 0, "Empty buffer len() not set to zero")
        self.rb.store((np.zeros(1),np.zeros(1),0,np.zeros(1)))
        self.assertEqual(len(self.rb), 1, "Partially full buffer len() not returning number of items")
        for _ in range(self.cap+1):
            self.rb.store((np.zeros(1),np.zeros(1),0,np.zeros(1)))
        self.assertEqual(len(self.rb), self.cap, "Full buffer len() not returning capacity")

    # Test sample method
    def test_sample(self):
        # Underfull Buffer Sampling
        self.rb.store((np.zeros(1),np.zeros(1),0,np.zeros(1)))
        obs, act, rwd, nob = self.rb.sample() # State (observation), action, reward, next state
        self.assertEqual(obs.shape[0], self.bs, "Sampled OBS not equal to batch size") # data duplicated to fill batch for underfull buffer
        self.assertEqual(act.shape[0], self.bs, "Sampled ACT not equal to batch size")
        self.assertEqual(rwd.shape[0], self.bs, "Sampled RWD not equal to batch size")
        self.assertEqual(nob.shape[0], self.bs, "Sampled NOB not equal to batch size")
        # Full Buffer Sampling
        for idx in range(self.cap+1):
            self.rb.store((idx*np.ones(1),idx*np.ones(1),idx,idx*np.ones(1)))
        obs, act, rwd, nob = self.rb.sample() # State (observation), action, reward, next state
        self.assertFalse(np.all(obs[:,0] == obs[0,0]), "Sampled values were all the same (Stochastically possible but unlikely)")
        # Buffer Sequential Sampling
        sequential = True
        sampled_rewards = np.transpose(rwd).flatten()
        val = sampled_rewards[0]
        for idx in range(1, len(sampled_rewards)):
            if not sampled_rewards[idx] == val + 1: sequential = False
            val = sampled_rewards[idx]
        self.assertFalse(sequential, "Sampled values were sequential (Stochastically possible but unlikely)")

    # Test Biased sample method
    def test_bias_sample(self):
        # Setup
        for idx in range(self.cap):
            self.rb.store((idx*np.ones(1),idx*np.ones(1),idx,idx*np.ones(1)))
        # Partial Bias
        for bias in range(1, self.bs): # 1 is 100% bias (all should be sequential)
            obs, act, rwd, nob = self.rb.biased_sample(bias)
            expect_seq = int(self.bs/bias)
            sequential = True
            sampled_rewards = np.transpose(rwd).flatten()
            val = sampled_rewards[0]
            for idx in range(1, expect_seq):
                if not sampled_rewards[idx] == val + 1: sequential = False
                val = sampled_rewards[idx]
            self.assertTrue(sequential, "Partial bias (%s) sample values were not sequential" % bias)
        # Wrapped Bias
        for _ in range(int(self.bs/2)):
            self.rb.store((np.ones(1),np.ones(1),1,np.ones(1)))
        obs, act, rwd, nob = self.rb.biased_sample(1)
        sampled_rewards = np.transpose(rwd).flatten()
        expect = [1 for _ in range(int(self.bs/2))]
        for x in reversed(range(self.bs-int(self.bs/2))):
            expect.append(self.cap-x-1)
        self.assertTrue(np.all(sampled_rewards == np.array(expect)), "100% Bias with capacity wrap not valid")

    # Runs after each test
    def tearDown(self):
        pass



################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("File meant for use as module, executing unit tests")
    unittest.main()
