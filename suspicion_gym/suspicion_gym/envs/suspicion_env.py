import gym

import numpy as np

import time
from tkinter import *
from tkinter import font
from PIL import Image,ImageTk

class SuspicionEnv(gym.Env):
    def __init__(self, num_players, gui_size=None, gui_delay=0):
        super(SuspicionEnv, self).__init__()
        # Game Options
        self.__dynSus_numCharacters = 10
        self.__dynSus_charNames = None
        self.__dynSus_startGemCount = None # Follows game rules unless specified
        self.__dynSus_boardWidth = 4
        self.__dynSus_boardHeight = 3
        # Paremterized Options
        self.__num_players = num_players
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
        self.__state = None
        # Return
        return self.__state

    def observe(self):
        return self.__player_turn, self.__state

    def step(self, action):
        # Setup
        # Validate Action
        if not self.action_space.contains(action):
            raise Exception("Invalid Action (%s)" % str(action)) # error out -> didnt meet act space rules
        elif True: # func to validate act choice based on state
            ### Todo: Counter and error if same bad action given N times?
            raise Exception("Invalid Action (%s)" % str(action)) # error out -> didnt meet act space rules
            return self.__state, -1, False, {} # Return negative reward, do not update state or turn, force agent to repick and learn
        # Perform Action
        obs, reward, done = None, None, None
        info = {}
        self.__player_turn = self.__player_turn + 1 if self.__player_turn < self.__num_players else 0
        # Return
        return obs, reward, done, info

    def render(self, mode="human",):
    	if mode == "human":
            # Validate
            if self.gui is not None:
                # Setup
                # Draw
                # Human Viewing Delay
                time.sleep(self.gui_delay)

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
            self.__dynSus_charNames = ("Buford Barnswallow", "Dr. Ashraf Najem", "Earl of Volesworthy", "Lily Nesbitt", "Mildred Wellington", "Nadia Bwalya", "Remy La Rocque", "Stefano Laconi", "Trudie Mudge", "Viola Chung")
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
        action_card_limits.append(self.__num_players) # player index for question cards
        action_card_limits.append(self.__dynSus_numCharacters) # trapdoor character
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
        obs_limits.append(6) # die1 roll
        obs_limits.append(6) # die2 roll
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
