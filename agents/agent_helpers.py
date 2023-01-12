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
-> room_gems: array of gems available in players characters current room (R,G,Y)
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
    state_len -= (1 + 3 + (2*num_characters) + 2 + 3 + (2*16)) # Subtract non player specific
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

    room_gems = state[state_idx:state_idx+3]
    state_idx += 3

    for card in range(2):
        action_cards.append(state[state_idx:state_idx+16])
        state_idx += 16

    for opponent_idx in range(num_players-1):
        knowledge.append(state[state_idx:state_idx+num_characters])
        state_idx += num_characters

    # Return
    return num_players, invite_idx, bank_gems, player_gems, character_locations, die_rolls, room_gems, action_cards, knowledge

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
    # setup
    act_idx = 0
    # Get State info
    num_players, invite_idx, bank_gems, player_gems, char_locs, die_rolls, room_gems, act_cards, knowledge = decode_state(state, num_characters)
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
-> No return
"""
def randActComp_dieMove(action, die_rolls, character_locations, num_characters=10, board_width=4, board_height=3):
    # Setup
    act_idx = 0 # Start updating at beginning of action array
    die_rolls_charOnly = list(map(lambda x: np.random.randint(0, num_characters) if x == num_characters else x, list(die_rolls)))
    # Apply Roll
    for roll in die_rolls_charOnly:
        action[act_idx] = roll
        curr_loc = character_locations[roll]
        action[act_idx+1:act_idx+3] = curr_loc
        while True:
            x_or_y = np.random.randint(0, 2)
            plus_or_minus = 1 if np.random.randint(0, 2) == 1 else -1
            new_val = action[act_idx+1+x_or_y] + plus_or_minus
            if x_or_y == 0 and new_val >= 0 and new_val < board_width:
                break
            elif x_or_y == 1 and new_val >= 0 and new_val < board_height:
                break
        action[act_idx+1+x_or_y] += plus_or_minus
        character_locations[roll] = action[act_idx+1:act_idx+3]
        act_idx += 3

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
    act_idx = 6 # Start updating after die moves in action array
    # Decode State
    num_players, invite_idx, bank_gems, player_gems, char_locs, die_rolls, room_gems, act_cards, knowledge = decode_state(state, num_characters)
    # Action Card Selection
    if act_card_idx is None: act_card_idx = np.random.randint(0, 2) # random select one of two action cards
    if act_order is None: act_order = np.random.randint(0, 2) # random select to por bottom action to apply first
    card_act_idxs = np.where(act_cards[act_card_idx] == 1)[0]
    # Apply Action Card
    for act_card_action_idx in range(len(card_act_idxs)):
        # Setup
        act_card_action = card_act_idxs[act_card_action_idx] if act_order == 0 else card_act_idxs[len(card_act_idxs)-1-act_card_action_idx]
        # Set Action
        action[act_idx] = act_card_idx
        if act_card_action == 0: # Trapdoor action
            action[act_idx+7] = 1 + np.random.randint(0, num_characters)
            action[act_idx+8] = np.random.randint(0, board_width)
            action[act_idx+9] = np.random.randint(0, board_height)
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
            action[act_idx+6] = np.random.randint(1, num_players) # 0 indicates self (no ask), all others are opponent offset from current player
        # Increment act idx
        act_idx += 10

################################################################################
################################################################################
## Classes
################################################################################
################################################################################



################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("No executable code, meant for use as module only")
