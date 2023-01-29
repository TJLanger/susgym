#!/usr/bin/env python

################################################################################
################################################################################
## Authorship
################################################################################
################################################################################
##### Project
############### susgym
##### Name
############### suspicion_env.py
##### Author
############### Trenton Langer
##### Creation Date
############### 20221117
##### Description
############### Suspicion environment code
##### Project
############### YYYYMMDD - (CONTRIBUTOR) EXAMPLE



################################################################################
################################################################################
## Imports
################################################################################
################################################################################
# OpenAI gym
import gym

# Other
import itertools
import math
import numpy as np
import random
import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple, Union

# Graphics
from suspicion_gym.suspicion_gym.guis import basic_gui



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
"""
description:
-> Suspicion game environment, inheriting from OpenAI gym base class
"""
class SuspicionEnv(gym.Env):
    def __init__(self, num_players, gui_size=None, gui_delay=0, debug=False):
        super(SuspicionEnv, self).__init__()
        # ENV Options
        self.__partialReward = True
        # Game Options
        self.__dynSus_numCharacters = 10
        self.__dynSus_charNames = None
        self.__dynSus_startGemCount = None # Follows game rules unless specified
        self.__dynSus_boardWidth = 4
        self.__dynSus_boardHeight = 3
        # Paremterized Options
        self.__num_players = num_players
        self.__debug = debug
        # OpenAI Gym Setup
        ###self.action_space = gym.spaces.MultiDiscrete(self._gen_act_limits())
        self.action_space = SuspicionActionSpace(self._gen_act_limits())
        self.observation_space = gym.spaces.MultiDiscrete(self._gen_obs_limits(), dtype=np.int8)
        self.gui = None
        self.gui_delay = gui_delay
        if gui_size is not None:
            self.gui = basic_gui.SusGui(num_players, gui_width=gui_size, gui_height=gui_size, max_gems=self._gen_max_gems())
        # Reset
        self.reset() # Most init code in reset, just call function to avoid copy/paste

    def reset(self):
        # Suspicion Setup
        self.__charAssigns, self.__inviteDeck = self._init_characters()
        self.__board = self._init_board()
        self.__cards = self._init_cards()
        self.__discards = []
        # OpenAI Gym Setup
        if self.gui is not None:
            self.gui.update()
        # Agent Setup
        self.__agent_observations = [None for i in range(self.__num_players)]
        self.__agent_rewards = [0 for i in range(self.__num_players)]
        self.__agent_dones = [False for i in range(self.__num_players)]
        self.__agent_infos = [None for i in range(self.__num_players)]
        self.__agent_cards = [[] for i in range(self.__num_players)]
        self.__agent_kbs = np.ones((self.__num_players,self.__num_players-1,self.__dynSus_numCharacters), dtype=np.int8)
        for i in range(self.__num_players):
            # Draw Cards
            card_idx = np.random.randint(0, len(self.__cards))
            self.__agent_cards[i].append(self.__cards.pop(card_idx))
            card_idx = np.random.randint(0, len(self.__cards))
            self.__agent_cards[i].append(self.__cards.pop(card_idx))
            # Update Knowledge Base
            self.__agent_kbs[i,:,self.__charAssigns[i]] = 0
        self.__agent_cards = np.array(self.__agent_cards)
        # State Init
        self.__player_turn = np.random.randint(0, self.__num_players)
        self.__state = self._init_state()
        self.render()
        # Return
        return self.__state

    def observe(self):
        # Update for specific player before returning
        self._personalize_state(self.__state)
        # Return
        return self.__player_turn, self.__state.copy()

    def step(self, action):
        # Setup
        # Validate Action
        if not self.action_space.contains(action):
            raise Exception("Out of Space Action (%s)" % str(action)) # error out -> didnt meet act space rules
        elif not self._validate_action(action): # func to validate act choice based on state
            ### Todo: Counter and error if same bad action given N times?
            return self.__state, -10, False, {} # Return negative reward, do not update state or turn, force agent to repick and learn
            ###raise Exception("Invalid Action (%s)" % str(action)) # error out -> didnt meet act space rules
        # Perform Action
        reward, done = self._apply_action(action) # Also modifies state (in place)
        self._personalize_state(self.__state) # Update state with new cards/knowledge (for render - prior to turn update)
        # Other ENV Updates
        self.__player_turn = self.__player_turn + 1 if self.__player_turn < self.__num_players - 1 else 0
        state_idx = 1 + 3 + 3*self.__num_players + 2*self.__dynSus_numCharacters # Invite idx, bank/player gems, character locations, then die rolls
        for die_num in range(2):
            if die_num == 0:
                roll = np.random.randint(0, math.floor(self.__dynSus_numCharacters/2)+1) # standard 1-> 6 roll, need to lookup chars
                self.__state[state_idx] = self.__die1_lup[roll]
            else:
                roll = np.random.randint(0, math.ceil(self.__dynSus_numCharacters/2)+1) # standard 1-> 6 roll, need to lookup chars
                self.__state[state_idx] = self.__die2_lup[roll]
            state_idx += 1
        # Return
        info = {}
        return self.__state.copy(), reward, done, info

    def render(self, mode="human",):
        self._personalize_state(self.__state)
        if mode == "human":
            # Validate
            if self.gui is not None:
                # Setup
                gem_start, gem_end = 1, 1+3*(self.__num_players+1)
                gem_counts = self.__state[gem_start:gem_end]
                char_locs = self.__state[gem_end:gem_end+2*self.__dynSus_numCharacters]
                # Draw
                self.gui.draw(gem_counts, char_locs)
                # Human Viewing Delay
                time.sleep(self.gui_delay)
            elif self.__debug:
                dbg_idx = 0
                print("Players: (%s)" % str(list(range(self.__num_players))))
                print("Turn: %s" % str(self.__player_turn))
                print("Invite Deck Index: %s" % str(self.__state[dbg_idx]))
                dbg_idx += 1
                print("Remaining gems: %s" % str(self.__state[dbg_idx:dbg_idx+3]))
                dbg_idx += 3
                for player_idx in range(self.__num_players):
                    print("Player %s:" % str(player_idx))
                    print("\tGems: %s" % str(self.__state[dbg_idx:dbg_idx+3]))
                    print("\tCards: %s" % str(self.__agent_cards[player_idx]))
                    player_char = self.__charAssigns[player_idx]
                    print("\tCharacter: (%s, %s)" % (str(player_char),str(self.__dynSus_charNames[player_char])))
                    dbg_idx += 3
                print("Character locations:")
                for char_idx in range(self.__dynSus_numCharacters):
                    tabs = "\t" if len(str(self.__dynSus_charNames[char_idx])) > 17 else "\t\t"
                    print("\t(%s, %s):%s%s" % (str(char_idx),str(self.__dynSus_charNames[char_idx]),tabs,self.__state[dbg_idx:dbg_idx+2]))
                    dbg_idx += 2
                print("Die Rolls: (%s)" % str(list(map(lambda x: str(x)+":"+self.__dynSus_charNames[x], list(self.__state[dbg_idx:dbg_idx+2])))))
                dbg_idx += 2
                print("Player Specific State:")
                print("\tPlayer Char: %s" % self.__state[dbg_idx])
                dbg_idx += 1
                print("\tCards: (%s, %s)" % (self.__state[dbg_idx:dbg_idx+16], self.__state[dbg_idx+16:dbg_idx+32]))
                dbg_idx += 32
                print("\tKnowledge:")
                while dbg_idx < len(self.__state):
                    print("\t\tOpp: %s" % (str(self.__state[dbg_idx:dbg_idx+self.__dynSus_numCharacters])))
                    dbg_idx += self.__dynSus_numCharacters
                # print("\tALL Knowledge:")
                # for pidx in range(self.__num_players):
                #     print("\t\tPlayer: %s" % pidx)
                #     for opp_idx in range(self.__num_players-1):
                #         print("\t\t\tOpp: %s" % self.__agent_kbs[pidx][opp_idx])
                print("\n\n\n")

    def close(self):
        if self.gui is not None:
            self.gui.destroy()

    """
    description:
    -> Assigns a character for each player, and returns assignments and leftover characters
    parameters:
    -> num_characters (From susEnv object)
    -> num_players (From susEnv object)
    """
    def _init_characters(self):
        # Setup
        characters = []
        deck = list(range(self.__dynSus_numCharacters))
        if self.__dynSus_numCharacters == 10:
            self.__dynSus_charNames = ("Buford Barnswallow", "Dr. Ashraf Najem", "Earl of Volesworthy", "Lily Nesbitt", "Mildred Wellington", "Nadia Bwalya", "Remy La Rocque", "Stefano Laconi", "Trudie Mudge", "Viola Chung", "?")
            self.__die1_lup = (6,3,7,5,8,self.__dynSus_numCharacters)
            self.__die2_lup = (4,2,0,1,9,self.__dynSus_numCharacters)
        else:
            auto_names = list(map(lambda x: "Char"+str(x), list(range(self.__dynSus_numCharacters))))
            auto_names.append("?")
            self.__dynSus_charNames = tuple(auto_names)
            die1 = list(range(0, math.floor(self.__dynSus_numCharacters/2)))
            die1.append(self.__dynSus_numCharacters)
            self.__die1_lup = tuple(die1)
            die2 = list(range(math.floor(self.__dynSus_numCharacters/2), self.__dynSus_numCharacters))
            die2.append(self.__dynSus_numCharacters)
            self.__die2_lup = tuple(die2)
        # Assign Characters
        for player_idx in range(self.__num_players):
            char_idx = np.random.randint(0, len(deck))
            characters.append(deck.pop(char_idx))
        # Shuffle deck
        np.random.shuffle(deck)
        # Return
        return characters, deck

    """
    description:
    -> Creates a Suspicion game board, representing gems available in each room
    parameters:
    -> board_width (From susEnv object)
    -> board_height (From susEnv object)
    return:
    -> Numpy tensor of shape room width x room height x 3 gem types
    """
    def _init_board(self):
        # Setup
        board = np.zeros((self.__dynSus_boardWidth,self.__dynSus_boardHeight,3), dtype=np.int8)
        # Mark Available gems
        if self.__dynSus_boardWidth == 4 and self.__dynSus_boardHeight == 3:
            # Shape of X, Y, [Red, Green, Yellow]
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
        else:
            pass # Todo: Create random method for generating non-standard board sizes
        # Return
        return board

    """
    description:
    -> Creates list of Suspicion cards and assigns to each player
    parameters:
    -> num_players (From susEnv object)
    return:
    -> Numpy tensor of shape num_cards x 16 card actions
    """
    def _init_cards(self):
        # Setup
        cards = []
        # Create Cards
        ### One hot array of possible actions (2 per card)
        ### trapdoor getRedGem getGreen getYellow getRoom viewDeck 10xaskCharacter (Alphabetical)
        cards.append([0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0]) # getYellow, askRemy
        cards.append([0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]) # getRoom, viewDeck
        cards.append([0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0]) # getRed, askNadia
        cards.append([0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]) # getGreen, askLily
        cards.append([0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0]) # viewDeck, askBuford
        cards.append([0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0]) # getRed, askEarl
        cards.append([0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0]) # getRoom, askNadia
        cards.append([0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0]) # getGreen, askStefano
        cards.append([0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0]) # getYellow, viewDeck
        cards.append([0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0]) # getRoom, askDrAshraf
        cards.append([0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0]) # getGreen, viewDeck
        cards.append([0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0]) # getRed, viewDeck
        cards.append([0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0]) # getRoom, askMildred
        cards.append([1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]) # getRoom, trapdoor
        cards.append([0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0]) # getRoom, askEarl
        cards.append([0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]) # getRoom, askRemy
        cards.append([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1]) # viewDeck, askViola
        cards.append([0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0]) # getRoom, askStefano
        cards.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]) # getRoom, askViola
        cards.append([0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]) # getRoom, viewDeck
        cards.append([0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0]) # getRoom, askLily
        cards.append([0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0]) # getYellow, askMildred
        cards.append([0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0]) # getRoom, askBuford
        cards.append([1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]) # getRoom, trapdoor
        cards.append([1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]) # trapdoor, askDrAshraf
        cards.append([0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]) # getRoom, viewDeck
        cards.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0]) # getRoom, askTrudie
        cards.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]) # trapdoor, askTrudie
        # Shuffle
        np.random.shuffle(cards)
        # Return
        return cards

    """
    description:
    -> Calculates initial gem count based on game settings
    parameters:
    -> num_players (From susEnv object)
    return:
    -> max number of gems based on game settings
    """
    def _gen_max_gems(self):
        num_gems = None
        if self.__dynSus_startGemCount is not None:
            num_gems = self.__dynSus_startGemCount # Allow custom override
        elif self.__num_players < 4:
            num_gems = 5 # Suspicion rules
        elif self.__num_players < 5:
            num_gems = 9 # Suspicion rules
        elif self.__num_players < 7:
            num_gems = 12 # Suspicion rules
        else:
            num_gems = 2 * self.__num_players # safety case for custom games without specified gem
        return num_gems

    """
    description:
    -> Creates array of valid action space values for use in MultiDiscrete space
    parameters:
    -> num_characters (From susEnv object)
    -> num_players (From susEnv object)
    -> board_width (From susEnv object)
    -> board_height (From susEnv object)
    return:
    -> array of action space ranges
    """
    def _gen_act_limits(self):
        # Setup
        act_limits = []
        td_target = self.__dynSus_numCharacters * self.__dynSus_boardWidth * self.__dynSus_boardHeight # Trapdoor = any char to any room
        # Generate limits
        act_limits.append(self.__dynSus_numCharacters) # Die 1 move character
        act_limits.append(4) # Die 1 move (up, down, right, left)
        act_limits.append(self.__dynSus_numCharacters) # Die 2 move character
        act_limits.append(4) # Die 2 move (up, down, right, left)
        act_limits.append(2) # Action card select
        act_limits.append(2) # Action order select
        act_limits.append(td_target) # Action 1 target (trapdoor can only be action 1)
        act_limits.append(6) # Action 2 target (Gem take - 3 color lucky/room, view deck, character ask - up to number of opponents)
        for opponent in range(self.__num_players-1):
            act_limits.append(self.__dynSus_numCharacters) # Endgame character guesses
        # Return
        return np.array(act_limits)

    """
    description:
    -> Creates array of valid observation space values for use in MultiDiscrete space
    parameters:
    -> num_characters (From susEnv object)
    -> num_players (From susEnv object)
    -> board_width (From susEnv object)
    -> board_height (From susEnv object)
    return:
    -> array of observation space ranges
    """
    def _gen_obs_limits(self):
        # Setup
        obs_limits = []
        # Generate limits
        ###obs_limits.append(self.__num_players) # turn indicator
        obs_limits.append(self.__dynSus_numCharacters-self.__num_players) # invite deck size
        for player_idx in range(self.__num_players+1): # player and bank gem counts
            obs_limits.append(self._gen_max_gems()+1) # red gem count
            obs_limits.append(self._gen_max_gems()+1) # green gem count
            obs_limits.append(self._gen_max_gems()+1) # yellow gem count
        for char_idx in range(self.__dynSus_numCharacters): # character locations x/y
            obs_limits.append(self.__dynSus_boardWidth)
            obs_limits.append(self.__dynSus_boardHeight)
        obs_limits.append(self.__dynSus_numCharacters+1) # die1 roll, num characters + '?' possible
        obs_limits.append(self.__dynSus_numCharacters+1) # die2 roll
        obs_limits.append(self.__dynSus_numCharacters) # player character identity
        for action_idx in range(2*16): # action card 1hot flags, 2 cards x 16 flags
            obs_limits.append(2)
        for opponent_idx in range(self.__num_players-1): # Knowledge bases for each opponent
            for char_idx in range(self.__dynSus_numCharacters):
                obs_limits.append(2) # 1hot for each possible character
        # Return
        return np.array(obs_limits)

    """
    description:
    -> Creates a starting valid state
    parameters:
    -> num_characters (From susEnv object)
    -> num_players (From susEnv object)
    -> board_width (From susEnv object)
    -> board_height (From susEnv object)
    return:
    -> Numpy tensor representing a valid initial board state
    """
    def _init_state(self):
        # Setup
        idx = 0 # variable to help set state values in loops/etc
        state = np.zeros(self.observation_space.shape, dtype=np.int8)
        char_x, char_y = 0, 0
        # Set Values
        ### state[idx] = self.__player_turn # turn indicator (disabled)
        ### idx += 1
        idx += 1 # invite deck index already set to 0, skip variable
        state[idx:idx+3] = self._gen_max_gems() # gem counts for the bank / starting piles
        idx += 3
        for player_idx in range(self.__num_players):
            idx += 3 # player gem counts already set to 0, skip variables
        for char_idx in itertools.chain(self.__charAssigns, self.__inviteDeck):
            # set state values
            state[idx+2*char_idx] = char_x
            state[idx+2*char_idx+1] = char_y
            # update iterators
            char_x = char_x + 1 if char_x < self.__dynSus_boardWidth - 1 else 0
            if char_y == 1 and char_x > 0: char_x = self.__dynSus_boardWidth - 1
            if char_x == 0: char_y += 1
        idx += 2*self.__dynSus_numCharacters
        for die_num in range(2):
            if die_num == 0:
                roll = np.random.randint(0, math.floor(self.__dynSus_numCharacters/2)+1) # standard 1-> 6 roll, need to lookup chars
                state[idx] = self.__die1_lup[roll]
            else:
                roll = np.random.randint(0, math.ceil(self.__dynSus_numCharacters/2)+1) # standard 1-> 6 roll, need to lookup chars
                state[idx] = self.__die2_lup[roll]
            idx += 1
        # Apply player specific info
        self._personalize_state(state)
        # return
        return state

    """
    description:
    -> apply player specific state info to a given state reference
    parameters:
    -> num_characters (From susEnv object)
    -> num_players (From susEnv object)
    -> board_width (From susEnv object)
    -> board_height (From susEnv object)
    return:
    -> No return. In place modification of given state reference
    """
    def _personalize_state(self, state):# TODO: validate against board/stat
        # Setup
        player_state = []
        player_char = self.__charAssigns[self.__player_turn]
        # Character Identity
        player_state.append(player_char)
        # Action Cards
        for card in self.__agent_cards[self.__player_turn]:
            player_state.extend(card)
        # Knowledge Base
        for kb in self.__agent_kbs[self.__player_turn]:
            player_state.extend(kb)
        # Modification
        state[-len(player_state):] = np.array(player_state, dtype=np.int8)

    """
    description:
    -> checks if a given action is valid for the current game state
    parameters:
    -> state (From susEnv object)
    -> action
    return:
    -> isValid, boolean indicating if action was valid or not
    """
    def _validate_action(self, action):
        # setup
        act_idx = 0
        state_idx = 1 # skip invite idx in state, start at gem counts
        act_msg = False
        ### state_idx = 2 # bigger skip if using turn indicator in state
        # Get State info
        bank_gems = self.__state[state_idx:state_idx+3]
        for player_idx in range(self.__num_players+1):
            state_idx += 3 # increment past gem counts for bank and all players
        char_locs = np.copy(self.__state[state_idx:state_idx+2*self.__dynSus_numCharacters]) # Copy to prevent updates by reference
        state_idx += 2*self.__dynSus_numCharacters
        die_rolls = self.__state[state_idx:state_idx+2]
        state_idx += 2
        state_idx += 1 # Skip character identity, already known in ENV
        act_cards = []
        for card in range(2):
            act_cards.append(self.__state[state_idx:state_idx+16])
            state_idx += 16
        ###print("Act Cards: %s" % str(act_cards))
        # Check if normal gameplay or endgame (guessing identities)
        if np.all(bank_gems > 0):
            # Check Die Moves
            char_loc_start = 1 + 3*(self.__num_players+1)
            for roll in die_rolls:
                # Setup
                die_move = action[act_idx:act_idx+2] # char, direction (up, down, right, left)
                move_dir = ([1,0] if die_move[1] % 2 == 0 else [-1,0]) if die_move[1] > 1 else ([0,1] if die_move[1] % 2 == 0 else [0,-1])
                act_idx += 2
                # Check Character
                if die_move[0] != roll and roll != self.__dynSus_numCharacters:
                    if act_msg: print("INVALID ACT: Attempting to move Char %s on die roll %s" % (die_move[0], roll))
                    return False # Fail if character doesnt match, unless rolled a '?'
                # Check Move
                move_char = die_move[0] if roll == self.__dynSus_numCharacters else roll
                new_loc = char_locs[2*move_char:2*move_char+2] + move_dir
                if new_loc[0] >= 0 and new_loc[0] < self.__dynSus_boardWidth and new_loc[1] >= 0 and new_loc[1] < self.__dynSus_boardHeight:
                    char_locs[2*move_char:2*move_char+2] += move_dir
                else:
                    if act_msg: print("INVALID ACT: Attempting to move Char off board (%s)" % die_move)
                    return False
            # Check Action Card Actions
            ### Any card select / card order / target combo that meets the action space requirements is valid
            ### EXCEPT - need to validate gem takes from room (after die rolls and potential trapdoors)
            act_card_select, act_card_order = action[act_idx:act_idx+2]
            act_idx += 2
            act_card_start = char_loc_start + 2*self.__dynSus_numCharacters + 2
            act_card_actions = np.where(self.__agent_cards[self.__player_turn][act_card_select] == 1)[0]
            if act_card_order == 1:
                act_card_actions = np.flip(act_card_actions)
            for act_card_action in act_card_actions:
                # Setup
                act_card_target = action[act_idx+act_card_order]
                target_max = 120 if act_card_order == 0 else 6
                # Apply action
                if act_card_action == 0: # Trapdoor
                    td_char = int(act_card_target / (self.__dynSus_boardWidth * self.__dynSus_boardHeight))
                    td_room = act_card_target % (self.__dynSus_boardWidth * self.__dynSus_boardHeight)
                    td_roomx = int(td_room / self.__dynSus_boardHeight)
                    td_roomy =  td_room % self.__dynSus_boardHeight
                    char_locs[2*td_char:2*td_char+2] = [td_roomx, td_roomy]
                elif act_card_action == 4: # Gem Take (Room)
                    # Take Gem
                    gem_idx = int(3*act_card_target/target_max)
                    player_char = self.__charAssigns[self.__player_turn]
                    room_gems = self.__board[char_locs[2*player_char]][char_locs[2*player_char+1]]
                    if not room_gems[gem_idx] == 1:
                        if act_msg: print("INVALID GEM:\tTurn\t(%s)\tPC\t(%s)\tRoom\t(%s)\tTry\t(%s)" % (self.__player_turn, player_char, room_gems, gem_idx))
                        return False
                # Update Iteration Idx
                act_card_order = 1 - act_card_order # Update order to pick other target for next action        else: # Only need identity guesses
            pass # Any identity guess that meets the action space requirements is valid
        # Return Valid
        return True

    """
    description:
    -> applies action (already validated) to internal state
    parameters:
    -> state (From susEnv object)
    -> action
    return:
    -> reward, any float value associated with reard for the current action
    -> isDone, boolean indicating if action ended the game or not
    """
    def _apply_action(self, action):
        # Setup
        reward = 0 # TODO: investigate in progress rewards vs endgame only rewards?
        isDone = False
        act_idx = 0
        # Apply Action
        if np.all(self.__state[1:4] > 0): # Check if normal gameplay or endgame (guessing identities)
            # Move for Die Rolls
            char_loc_start = 1 + 3*(self.__num_players+1)
            for roll in self.__state[char_loc_start+2*self.__dynSus_numCharacters:char_loc_start+2*self.__dynSus_numCharacters+2]:
                # Setup
                die_move = action[act_idx:act_idx+2] # char, direction (up, down, right, left)
                move_dir = ([1,0] if die_move[1] % 2 == 0 else [-1,0]) if die_move[1] > 1 else ([0,1] if die_move[1] % 2 == 0 else [0,-1])
                move_char = die_move[0] if roll == self.__dynSus_numCharacters else roll
                # Update Char Locs
                self.__state[char_loc_start+2*move_char:char_loc_start+2*move_char+2] += move_dir
                # Update Iteration Idx
                act_idx += 2
            # Apply Actions
            act_card_select, act_card_order = action[act_idx:act_idx+2]
            act_idx += 2
            act_card_start = char_loc_start + 2*self.__dynSus_numCharacters + 2
            act_card_actions = np.where(self.__agent_cards[self.__player_turn][act_card_select] == 1)[0]
            if act_card_order == 1:
                act_card_actions = np.flip(act_card_actions)
            for act_card_action in act_card_actions:
                # Setup
                act_card_target = action[act_idx+act_card_order]
                target_max = 120 if act_card_order == 0 else 6
                # Apply action
                if act_card_action == 0: # Trapdoor
                    td_char = int(act_card_target / (self.__dynSus_boardWidth * self.__dynSus_boardHeight))
                    td_room = act_card_target % (self.__dynSus_boardWidth * self.__dynSus_boardHeight)
                    td_roomx = int(td_room / self.__dynSus_boardHeight)
                    td_roomy =  td_room % self.__dynSus_boardHeight
                    self.__state[char_loc_start+2*td_char:char_loc_start+2*td_char+2] = [td_roomx, td_roomy]
                elif act_card_action <= 3: # Gem Take (R, G, Y)
                    # Setup
                    pgems = self.__state[1+3*(1+self.__player_turn):1+3*(1+self.__player_turn)+3]
                    pgem_min = min(pgems)
                    pgem_score = 2*3*pgem_min + sum(pgems-pgem_min)
                    # Take Gem
                    gem_idx = act_card_action - 1
                    self.__state[1+gem_idx] -= 1 # lower bank count
                    self.__state[1+3*(1+self.__player_turn)+gem_idx] += 1 # add to player gem count
                    # Partial Reward
                    if self.__partialReward:
                        new_pgem_min = min(pgems)
                        new_pgem_score = 2*3*new_pgem_min + sum(pgems-new_pgem_min)
                        reward += new_pgem_score - pgem_score
                elif act_card_action == 4: # Gem Take (Room)
                    # Setup
                    pgems = self.__state[1+3*(1+self.__player_turn):1+3*(1+self.__player_turn)+3]
                    pgem_min = min(pgems)
                    pgem_score = 2*3*pgem_min + sum(pgems-pgem_min)
                    # Take Gem
                    gem_idx = int(3*act_card_target/target_max)
                    self.__state[1+gem_idx] -= 1 # lower bank count
                    self.__state[1+3*(1+self.__player_turn)+gem_idx] += 1 # add to player gem count
                    # Partial Reward
                    if self.__partialReward:
                        new_pgem_min = min(pgems)
                        new_pgem_score = 2*3*new_pgem_min + sum(pgems-new_pgem_min)
                        reward += new_pgem_score - pgem_score
                    # Update KBs
                    for char_idx in range(self.__dynSus_numCharacters):
                        char_loc = self.__state[char_loc_start+2*char_idx:char_loc_start+2*char_idx+2]
                        char_room_gems = self.__board[char_loc[0]][char_loc[1]]
                        if char_room_gems[gem_idx] == 1: continue
                        for player_idx in range(0, self.__num_players):
                            if player_idx == self.__player_turn: continue
                            update_idx = self.__player_turn - player_idx - 1 if self.__player_turn > player_idx else self.__num_players - 1 - player_idx + self.__player_turn
                            if update_idx >= self.__num_players:
                                update_idx -= self.__num_players
                            if self.__partialReward and self.__agent_kbs[player_idx][update_idx][char_idx] == 1:
                                reward -= 7/self.__dynSus_numCharacters
                            self.__agent_kbs[player_idx][update_idx][char_idx] = 0
                elif act_card_action == 5: # Check for Invite Deck View
                    viewed_icard = self.__inviteDeck[self.__state[0]]
                    if self.__partialReward:
                        for opp_kb in self.__agent_kbs[self.__player_turn]:
                            if opp_kb[viewed_icard] == 1: reward += 7/self.__dynSus_numCharacters
                    self.__agent_kbs[self.__player_turn][:,viewed_icard] = 0
                    self.__state[0] = self.__state[0] + 1 if self.__state[0] < len(self.__inviteDeck) - 1 else 0
                elif act_card_action <= 15: # Check for Character View Ask
                    target_player_offset = int((self.__num_players-1)*act_card_target/target_max)
                    target_player = self.__player_turn + target_player_offset + 1 # TODO: skewed probability to select certain opponents
                    if target_player >= self.__num_players:
                        target_player -= self.__num_players # Wrap around offset idx
                    target_char = np.where(self.__agent_cards[self.__player_turn][act_card_select][6:] == 1)[0][0]
                    # "Ask" and apply to KB
                    target_player_char = self.__charAssigns[target_player]
                    can_see = False
                    if self.__state[char_loc_start+2*target_char] == self.__state[char_loc_start+2*target_player_char]:
                        can_see = True
                    elif self.__state[char_loc_start+2*target_char+1] == self.__state[char_loc_start+2*target_player_char+1]:
                        can_see = True
                    for char_idx in range(self.__dynSus_numCharacters):
                        char_can_see = False
                        if self.__state[char_loc_start+2*target_char] == self.__state[char_loc_start+2*char_idx]:
                            char_can_see = True
                        elif self.__state[char_loc_start+2*target_char+1] == self.__state[char_loc_start+2*char_idx+1]:
                            char_can_see = True
                        if can_see != char_can_see:
                            if self.__partialReward and self.__agent_kbs[self.__player_turn][target_player_offset][char_idx] == 1:
                                reward += 7/self.__dynSus_numCharacters
                            self.__agent_kbs[self.__player_turn][target_player_offset][char_idx] = 0 # Not 'target_p' but just player offset
                # Update Iteration Idx
                act_card_order = 1 - act_card_order # Update order to pick other target for next action
            act_idx += 2
            # Draw and Replace Used Act Card
            self.__discards.append(self.__agent_cards[self.__player_turn][act_card_select].copy())
            if len(self.__cards) == 0:
                while len(self.__discards) > 0:
                    self.__cards.append(self.__discards.pop())
                random.shuffle(self.__cards)
            self.__agent_cards[self.__player_turn][act_card_select] = self.__cards.pop(0) # Overwrite card details with new card
        else: # Check player identity guesses
            # Setup
            isDone = True # Agent in terminal state after guessing
            # Check Guesses
            for opp_idx in range(self.__num_players, 1, -1):
                # Setup
                act_offset = 0-opp_idx
                char_guess = action[act_offset]
                opponent = self.__player_turn + opp_idx
                if opponent >= self.__num_players:
                    opponent -= self.__num_players # Wrap around offset idx
                # Check Guess
                if char_guess == self.__charAssigns[opponent]:
                    reward += 7 # 7 points per correct guess
            # Gem Points
            player_gems = self.__state[1+3*(1+self.__player_turn):1+3*(1+self.__player_turn)+3]
            while np.all(player_gems > 0):
                reward += 6 # 6 points per gem set
                player_gems = player_gems - 1
            reward += 1 * np.sum(player_gems) # 1 point per remaining gem
        # Return
        return reward, isDone

"""
description:
-> MultiDiscrete action space with game specific functionality
"""
class SuspicionActionSpace(gym.spaces.MultiDiscrete):
    def __init__(self, nvec: Union[np.ndarray, list], dtype=np.int64, seed: Optional[Union[int, np.random.Generator]] = None,):
        super().__init__(nvec, dtype, seed)

    @property
    def shape(self) -> Tuple[int, ...]:
        return super().shape

    @property
    def is_np_flattenable(self):
        return super().is_np_flattenable

    def sample(self, mask: Optional[tuple] = None) -> np.ndarray:
        # Todo: Pass state in mask variable, and generate mask based on state?
        return super().sample(mask) if mask is not None else super().sample()

    def contains(self, x) -> bool:
        return super().contains(x)

    def to_jsonable(self, sample_n: Iterable[np.ndarray]):
        return super().to_jsonable(sample_n)

    def from_jsonable(self, sample_n):
        return super().from_jsonable(sample_n)

    def __repr__(self):
        return "Suspicion(" + super().__repr__() + ")"

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()

    def __eq__(self, other):
        return super().__eq__(other)



################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("No executable code, meant for use as module only")
