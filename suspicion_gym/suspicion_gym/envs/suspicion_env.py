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
        self.gui = None
        self.gui_delay = gui_delay
        if gui_size is not None:
            pass
        # Reset
        self.reset() # Most init code in reset, just call function to avoid copy/paste

    def reset(self):
        # Suspicion Setup
        self.__charAssigns, self.__inviteDeck = self._init_characters()
        self.__board = self._init_board()
        self.__cards = self._init_cards()
        # OpenAI Gym Setup
        self.action_space = None
        self.observation_space = None
        if self.gui is not None:
            self.gui.update()
        # Agent Setup
        # State Init
        # Return
        return self.__state

    def step(self, action):
        # Setup
        # Validate Action
        if not self.action_space.contains(action):
            # TODO: Apply negative reward, and return same state, so agent has to learn/repick action
            ### Counter and error if same bad action given N times?
            pass
        # Perform Action
        obs, reward, done = None, None, None
        info = {}
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

