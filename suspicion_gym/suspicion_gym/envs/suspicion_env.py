import gym

import itertools
import math
import numpy as np

import time
from tkinter import *
from tkinter import font
from PIL import Image,ImageTk

class SuspicionEnv(gym.Env):
    def __init__(self, num_players, gui_size=None, gui_delay=0, debug=False):
        super(SuspicionEnv, self).__init__()
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
        self.action_space = gym.spaces.MultiDiscrete(self._gen_act_limits())
        self.observation_space = gym.spaces.MultiDiscrete(self._gen_obs_limits())
        self.gui = None
        self.gui_delay = gui_delay
        if gui_size is not None:
            pass # todo: self.gui = susGui(gui_width=gui_size, gui_height=gui_size, num_players=num_players)
        # Reset
        self.reset() # Most init code in reset, just call function to avoid copy/paste

    def reset(self):
        # Suspicion Setup
        self.__charAssigns, self.__inviteDeck = self._init_characters()
        self.__board = self._init_board()
        self.__cards = self._init_cards()
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
        # State Init
        self.__player_turn = np.random.randint(0, self.__num_players)
        self.__state = self._init_state()
        self.render()
        # Return
        return self.__state

    def observe(self):
        self._personalize_state(self.__state) # Update for specific player before returning
        return self.__player_turn, self.__state.copy()

    def step(self, action):
        # Setup
        # Validate Action
        if not self.action_space.contains(action):
            raise Exception("Out of Space Action (%s)" % str(action)) # error out -> didnt meet act space rules
        elif not self._validate_action(action): # func to validate act choice based on state
            ### Todo: Counter and error if same bad action given N times?
            ###return self.__state, -1, False, {} # Return negative reward, do not update state or turn, force agent to repick and learn
            raise Exception("Invalid Action (%s)" % str(action)) # error out -> didnt meet act space rules
        # Perform Action
        reward, done = self._apply_action(action) # Also modifies state (in place)
        info = {}
        self.__player_turn = self.__player_turn + 1 if self.__player_turn < self.__num_players else 0
        # Return
        return self.__state.copy(), reward, done, info

    def render(self, mode="human",):
    	if mode == "human":
            # Validate
            if self.gui is not None:
                # Setup
                # Draw
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
                print("\tRoom Gems: %s" % self.__state[dbg_idx:dbg_idx+3])
                dbg_idx += 3
                print("\tCards: (%s, %s)" % (self.__state[dbg_idx:dbg_idx+16], self.__state[dbg_idx+16:dbg_idx+32]))
                dbg_idx += 32
                print("\tKnowledge:")
                while dbg_idx < len(self.__state):
                    print("\t\tOpp: %s" % (str(self.__state[dbg_idx:dbg_idx+self.__dynSus_numCharacters])))
                    dbg_idx += self.__dynSus_numCharacters
                print("\n\n\n")


    def cleanup(self):
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
    def _gen_act_limits(self):        # Setup
        act_limits = []
        action_card_limits = []
        # Generate limits
        act_limits.append(self.__dynSus_numCharacters) # Die1 character, new x, new y
        act_limits.append(self.__dynSus_boardWidth)
        act_limits.append(self.__dynSus_boardHeight)
        act_limits.append(self.__dynSus_numCharacters) # Die2 character, new x, new y
        act_limits.append(self.__dynSus_boardWidth)
        act_limits.append(self.__dynSus_boardHeight)
        action_card_limits.append(2) # action card setup. number of cards in hand
        action_card_limits.append(2) # red gem take
        action_card_limits.append(2) # green gem take
        action_card_limits.append(2) # yellow gem take
        action_card_limits.append(2) # taken from room flag
        action_card_limits.append(2) # view invite deck flag
        action_card_limits.append(self.__num_players) # player index for question cards (offset from own player index, 0 indicates no action)
        action_card_limits.append(self.__dynSus_numCharacters+1) # trapdoor character (offset from standard idx, 0 indicates no action)
        action_card_limits.append(self.__dynSus_boardWidth) # trapdoor new x
        action_card_limits.append(self.__dynSus_boardHeight) # trapdoor new y
        act_limits.extend(action_card_limits) # two action cards (in order)
        act_limits.extend(action_card_limits)
        for opponent_idx in range(self.__num_players-1):
            act_limits.append(self.__dynSus_numCharacters) # chracter guess per opponent
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
            obs_limits.append(self._gen_max_gems()) # red gem count
            obs_limits.append(self._gen_max_gems()) # green gem count
            obs_limits.append(self._gen_max_gems()) # yellow gem count
        for char_idx in range(self.__dynSus_numCharacters): # character locations x/y
            obs_limits.append(self.__dynSus_boardWidth)
            obs_limits.append(self.__dynSus_boardHeight)
        obs_limits.append(self.__dynSus_numCharacters+1) # die1 roll, num characters + '?' possible
        obs_limits.append(self.__dynSus_numCharacters+1) # die2 roll
        obs_limits.append(2) # red gem in room flag
        obs_limits.append(2) # green gem in room flag
        obs_limits.append(2) # yellow gem in room flag
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
        # Room Gems
        room_idx = 1 + 3 + 3*self.__num_players + 2*player_char
        room_gems = self.__board[state[room_idx]][state[room_idx+1]]
        player_state.extend(room_gems)
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
        ### state_idx = 2 # bigger skip if using turn indicator in state
        # Get State info
        bank_gems = self.__state[state_idx:state_idx+3]
        for player_idx in range(self.__num_players+1):
            state_idx += 3 # increment past gem counts for bank and all players
        char_locs = self.__state[state_idx:state_idx+2*self.__dynSus_numCharacters]
        state_idx += 2*self.__dynSus_numCharacters
        die_rolls = self.__state[state_idx:state_idx+2]
        state_idx += 2
        room_gems = self.__state[state_idx:state_idx+3]
        state_idx += 3
        act_cards = []
        for card in range(2):
            act_cards.append(self.__state[state_idx:state_idx+16])
            state_idx += 16
        print("Act Cards: %s" % str(act_cards))
        # Check if normal gameplay or endgame (guessing identities)
        if np.all(bank_gems > 0):
            # Check Die Moves
            for die in range(2):
                # Setup
                die_move = action[act_idx:act_idx+3] # char, new_x, new_y
                act_idx += 3
                # Check on board
                if die_move[1] < 0 or die_move[1] >= self.__dynSus_boardWidth or die_move[2] < 0 or die_move[2] >= self.__dynSus_boardHeight:
                    return False
                # Check Move
                if die_move[0] in die_rolls:
                    die_rolls = np.delete(die_rolls, np.where(die_rolls == die_move[0])[0][0])
                    if die_move[1] == char_locs[2*die_move[0]] and ((die_move[2] == char_locs[2*die_move[0]+1] - 1) or (die_move[2] == char_locs[2*die_move[0]+1] + 1)):
                        char_locs[2*die_move[0]:2*die_move[0]+2] = die_move[1:3] # update char_loc for "?" followed by regular roll
                    elif die_move[2] == char_locs[2*die_move[0]+1] and ((die_move[1] == char_locs[2*die_move[0]] - 1) or (die_move[1] == char_locs[2*die_move[0]] + 1)):
                        char_locs[2*die_move[0]:2*die_move[0]+2] = die_move[1:3] # update char_loc for "?" followed by regular roll
                    else:
                        return False
                elif self.__dynSus_numCharacters in die_rolls:
                    die_rolls = np.delete(die_rolls, np.where(die_rolls == self.__dynSus_numCharacters)[0][0])
                    if die_move[1] == char_locs[2*die_move[0]] and ((die_move[2] == char_locs[2*die_move[0]+1] - 1) or (die_move[2] == char_locs[2*die_move[0]+1] + 1)):
                        char_locs[2*die_move[0]:2*die_move[0]+2] = die_move[1:3] # update char_loc for "?" followed by regular roll
                    elif die_move[2] == char_locs[2*die_move[0]+1] and ((die_move[1] == char_locs[2*die_move[0]] - 1) or (die_move[1] == char_locs[2*die_move[0]] + 1)):
                        char_locs[2*die_move[0]:2*die_move[0]+2] = die_move[1:3] # update char_loc for "?" followed by regular roll
                    else:
                        return False
                else:
                    return False # attempted to move invalid character
            # Check Action Card Actions
            for idx in range(2):
                # Setup
                act_card_action = action[act_idx:act_idx+10]
                print("Act Card Action: %s" % act_card_action)
                act_idx += 10
                # Validate Gem take
                if np.any(act_card_action[1:5]):
                    gem_idx = np.where(act_card_action[1:4] == 1)[0][0]
                    if act_card_action[4] > 0 and not room_gems[gem_idx] == 1: # Taking from room
                        print("Bad Room Take")
                        return False
                    elif act_card_action[4] == 0 and not act_cards[act_card_action[0]][1+gem_idx] == 1: # lucky lift
                        print("Bad Lucky Take (%s)" % gem_idx)
                        return False
                # Validate Invite Deck View
                if act_card_action[5] == 1 and not act_cards[act_card_action[0]][5] == 1:
                    print("Bad Invite")
                    return False
                # Validate Question Player
                if act_card_action[6] != 0 and not np.any(act_cards[act_card_action[0]][-self.__dynSus_numCharacters:] == 1):
                    print("Bad Question")
                    return False
                # Validate trapdoor
                if act_card_action[7] > 0 and not act_cards[act_card_action[0]][0] == 1:
                    print("Bad Trapdoor")
                    return False
        else: # Only need identity guesses
            return False
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
        reward = 0 # TODO: investigate in progress rewards vs endgame only reawrds?
        act_idx = 0
        # Apply Action
        if np.all(self.__state[1:4] > 0): # Check if normal gameplay or endgame (guessing identities)
            # Move for Die Rolls
            char_loc_start = 1 + 3*(self.__num_players+1)
            for die_move in range(2):
                # Update Char Locs
                self.__state[char_loc_start+2*action[act_idx]:char_loc_start+2*action[act_idx]+2] = action[act_idx+1:act_idx+3]
                # Update Iteration Idx
                act_idx += 3
            # Apply Actions
            for idx in range(2):
                # Setup
                act_card_action = action[act_idx:act_idx+10]
                # Apply action
                if np.any(act_card_action[1:5]): # Check for gem takes
                    gem_idx = np.where(act_card_action[1:4] == 1)[0][0]
                    self.__state[1+gem_idx] -= 1 # lower bank count
                    self.__state[1+3*(1+self.__player_turn)+gem_idx] += 1 # add to player gem count
                if act_card_action[5] == 1: # Check for Invite Deck View
                    viewed_icard = self.__inviteDeck[self.__state[0]]
                    self.__agent_kbs[self.__player_turn][:,viewed_icard] = 0
                    self.__state[0] = self.__state[0] + 1 if self.__state[0] < len(self.__inviteDeck) - 1 else 0
                if act_card_action[6] != 0: # Check For Question Player
                    # Setup
                    target_player = self.__player_turn + act_card_action[6]
                    if target_player > self.__num_players:
                        target_player -= self.__num_players # Wrap around offset idx
                    target_char = np.where(self.__agent_cards[act_card_action[0]][6:] == 1)[0][0]
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
                            self.__agent_kbs[self.__player_turn][target_player][char_idx] = 0
                if act_card_action[7] > 0: # Check For trapdoor
                    target_char = act_card_action[7] - 1
                    self.__state[char_loc_start+2*target_char:char_loc_start+2*target_char+2] = act_card_action[8:10]
                # Update Iteration Idx
                act_idx += 10
            # Draw and Replace Used Act Card
            self.__cards.append(self.__agent_cards[self.__player_turn].pop(action[6]))
            self.__agent_cards[self.__player_turn].append(self.__cards.pop(0))
        else: # Check player identity guesses
            pass
        # Return
        isDone = False # TODO: Set to true after all players have guessed identities
        return reward, isDone
