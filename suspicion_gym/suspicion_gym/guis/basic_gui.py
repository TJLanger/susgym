#!/usr/bin/env python

################################################################################
################################################################################
## Authorship
################################################################################
################################################################################
##### Project
############### susgym
##### Name
############### basic_gui.py
##### Author
############### Trenton Langer
##### Creation Date
############### 20221204
##### Description
############### Basic graphical user interface for suspicion_env
##### Project
############### YYYYMMDD - (CONTRIBUTOR) EXAMPLE



################################################################################
################################################################################
## Imports
################################################################################
################################################################################

# Other
import numpy as np
import os

# Graphics
from tkinter import *
from tkinter import font
from PIL import Image,ImageTk



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
-> Basic gui for suspicion_env
"""
class SusGui():
    # Initialization
    def __init__(self, num_players, gui_width=500, gui_height=500, max_gems=None):
        # Window Setup
        self.window = Tk()
        self.window.title('Suspicion')
        self.canvas = Canvas(self.window, width=gui_width, height=gui_height)
        self.canvas.pack()
        self.image_path = os.path.dirname(os.path.abspath(__file__)) + "/images"
        # Canvas Element IDs
        self.images = []
        self.character_ids = [] # Only need to save IDs for elements that might be deleted or referenced later
        self.gem_ids = []
        # Suspicion Set Up
        self.num_players = num_players
        self.char_colors = ("#F7E63C","#ECAD37","#613069","#F0BDCA","#B1BB44","#FFFFFF","#BDBBBB","#B27B35","#337FB2","#CB3C33",) # Alphabetical characters
        self.max_gems = max_gems
        if self.max_gems is None:
            self.max_gems = 5 if self.num_players < 4 else 9
            if self.num_players > 4:
                self.max_gems = 12
        # GUI Components
        self.gui_width = gui_width
        self.gui_height = gui_height
        self.board_height = int((2/3)*self.gui_height)
        self.board_width = min(int((4/3)*self.board_height), int((3/4)*self.gui_width))
        self.line_width = 2
        self.room_width = int(self.board_width/4)
        self.room_height = int(self.board_height/3)
        self.gem_width = int(self.room_width/5)
        self.gem_height = int(self.room_height/5)
        self.guest_width = int((self.gui_width-self.board_width-self.line_width*6)/2) # One extra line_width
        self.guest_height = min(int(self.gui_height/5),int(2*self.guest_width))
        self.label_font = font.Font(self.window, family='Arial', size=12, weight='bold')

    # Generate window & block to listen for events
    def mainloop(self):
        self.window.mainloop()

    # Generate window, but do not block
    def update(self):
        self.window.update()

    # Close Window
    def destroy(self):
        self.window.destroy()

    # Generate GUI
    def draw(self, gem_counts, char_locs):
        # Clear Old
        self.canvas.delete("all")
        # Draw Components
        self._draw_board()
        self._draw_gems(gem_counts)
        self._draw_characters(char_locs)
        # Display Changes
        self.update()

    # Generate Blank Suspicion Board (With Gem Images)
    def _draw_board(self):
        # Draw Horizontal Lines
        start_y = self.line_width
        for i in range(4):
            line_id = self.canvas.create_line(self.line_width, start_y, self.board_width, start_y,
                                              fill="#000000", width=self.line_width)
            start_y += self.room_height
        # Draw Vertical Lines
        start_x = self.line_width
        for i in range(5):
            line_id = self.canvas.create_line(start_x, self.line_width, start_x, self.board_height,
                                              fill="#000000", width=self.line_width)
            start_x += self.room_width
        # Draw Gems
        gem_locs = [
            [{'r':1},      {'r':1,'y':1},{'y':1,'g':1},{'r':1,'g':1},],
            [{'y':1,'g':1},{'r':1,'g':1},{'r':1,'y':1},{'y':1},      ],
            [{'r':1,'y':1},{'g':1},      {'r':1,'g':1},{'y':1,'g':1},],
        ]
        for i in range(len(gem_locs)):
            for j in range(len(gem_locs[i])):
                # Setup
                num_gems = len(gem_locs[i][j])
                room_center_y = (i * self.room_height) + int(self.room_height/2) + int(self.gem_height/2)
                room_center_x = (j * self.room_width) + int(self.room_width/2)
                # Insert Gem
                start_x = room_center_x if num_gems == 1 else room_center_x - self.gem_width
                for gem in gem_locs[i][j].keys():
                    # Pick Image
                    if gem == 'r':
                        gem_image = Image.open(self.image_path+"/sus_redGem.png")
                    elif gem == 'y':
                        gem_image = Image.open(self.image_path+"/sus_yellowGem.png")
                    elif gem == 'g':
                        gem_image = Image.open(self.image_path+"/sus_greenGem.png")
                    resized_gem = ImageTk.PhotoImage(gem_image.resize((self.gem_width,self.gem_height), Image.ANTIALIAS))
                    self.canvas.create_image(start_x, room_center_y, anchor='sw', image=resized_gem)
                    self.images.append(resized_gem) # Need to save image references, or tk will overwrite them
                    start_x += self.gem_width
        # Draw Guest List
        guests = ["Buford Barnswallow", "Earl of Volesworthy", "Mildred Wellington", "Viola Chung",
                  "Dr Ashraf Najem", "Nadia Bwalya", "Remy La Rocque", "Lily Nesbitt", "Trudie Mudge", "Stefano Laconi"]
        guest_x_idx = 0
        guest_y_idx = 0
        for guest in guests:
            # Setup
            guest_name = guest.replace(" ", "")
            # Draw
            guest_image = Image.open(self.image_path+"/sus_" + guest_name + ".png")
            resized_guest = ImageTk.PhotoImage(guest_image.resize((self.guest_width,self.guest_height), Image.ANTIALIAS))
            self.canvas.create_image(self.board_width + self.line_width*6 + guest_x_idx*self.guest_width,
                                     self.guest_height*guest_y_idx, anchor='nw', image=resized_guest)
            self.images.append(resized_guest)
            # Loop Control
            guest_x_idx += 1
            if(guest_x_idx > 1):
                guest_x_idx = 0
                guest_y_idx += 1

    # Draw Gem Stacks
    def _draw_gems(self, gem_counts):
        # Setup
        gem_text_width = int(self.board_width/(self.num_players+1)) # Extra 1 for 'bank' of gems
        gem_text_height = int((1/6)*self.gui_height)
        displayed_players = { idx: 0 for idx in range(self.num_players) }
        self.label_font['size'] = 0 - gem_text_height
        gem_width = min(int(gem_text_width/7), int((self.gui_height - self.board_height - gem_text_height)/self.max_gems))
        colors = ["#FF0000", "#00FF00", "#FFFF00"] # Red green yellow
        # Determine Fontsize
        names = ["P" + str(x) for x in range(self.num_players)] # TODO: Pass agent names into GUI for display
        names.insert(0, "Bank")
        name_lens = [len(n) for n in names]
        max_name = names[np.argmax(name_lens)]
        self.label_font['size'] = 0 - gem_text_height
        size = self.label_font.actual("size")
        while size > 1 and self.label_font.measure(max_name) >= gem_text_width:
            size -= 1
            self.label_font.configure(size=size)
        # Draw Gem Area
        for player_idx in range(1+self.num_players):
            # Read Info
            player_gems = gem_counts[3*player_idx:3*player_idx+3]
            # Draw
            label = Label(self.window, text=names[player_idx], font=self.label_font) # TODO: width in font size units, convert to pixels? and center text
            label.place(x=player_idx*gem_text_width,y=self.gui_height-int(gem_text_height/2))
            self.gem_ids.append(label)
            gem_x = player_idx*gem_text_width + gem_width
            for color_idx in range(3):
                gem_y = self.gui_height-int(gem_text_height/2)-gem_width
                for x in range(player_gems[color_idx]):
                    self.canvas.create_rectangle(gem_x, gem_y, gem_x+gem_width, gem_y+gem_width,
                                                 fill=colors[color_idx], outline='')
                    gem_y -= int(1.25*gem_width)
                gem_x += int(2*gem_width)

    # Add Character Locations
    def _draw_characters(self, character_locations):
        # Setup
        char_width = int(self.room_width/9) # sized for 4 chars with spaces between in each room
        room_cnts = [[0,0,0] for x in range(4)]
        positions = {}
        for idx in range(4):
            x = self.line_width + int(2*(idx+1))*char_width
            y = self.line_width + char_width
            positions[idx] = [x, y]
            positions[idx+4] = [self.room_width-x, self.room_height-y]
        positions[8] = [self.line_width + char_width, int(self.room_height/2)]
        positions[9] = [self.room_width - (self.line_width + char_width), int(self.room_height/2)]
        # Draw
        for char_idx in range(int(len(character_locations)/2)):
            room_x = character_locations[2*char_idx]
            room_y = character_locations[2*char_idx+1]
            position = positions[room_cnts[room_x][room_y]]
            room_x_px = (room_x * self.room_width)
            room_y_px = (room_y * self.room_height)
            self.canvas.create_rectangle(room_x_px+position[0], room_y_px+position[1],
                                         room_x_px+position[0]+char_width, room_y_px+position[1]+char_width,
                                         fill=self.char_colors[char_idx], outline='')
            room_cnts[room_x][room_y] += 1

################################################################################
################################################################################
## Main
################################################################################
################################################################################
if __name__ == "__main__":
    print("No executable code, meant for use as module only")
