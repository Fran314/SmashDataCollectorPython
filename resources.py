import os
import numpy
import sys
from cv2 import cv2
from customizable import res_path

def readTSV(path):
    list_file = open(path, 'r')
    file_content = list_file.read()
    lines = []
    curr_line = ""
    for c in file_content:
        if(c != '\n'):
            curr_line += c
        else:
            lines.append(curr_line)
            curr_line = ""
    if(curr_line != ""):
        lines.append(curr_line)
    
    to_return = []
    for line in lines:
        values = []
        curr_val = ""
        for c in line:
            if(c != '\t'):
                curr_val += c
            else:
                values.append(curr_val)
                curr_val = ""
        if(curr_val != ""):
            values.append(curr_val)
        to_return.append(values)
    return to_return


MAX_PLAYERS = 4 # Max number of players


CHARACTER_INFOS = readTSV(os.path.join(res_path, "characters_info.tsv"))
CHARACTER_CLOSEUPS = []
characters_tile = cv2.cvtColor(cv2.imread(os.path.join(res_path, "character_references", "23closeups.png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
tile_height = int(characters_tile.shape[0] / len(CHARACTER_INFOS))
tile_width = int(characters_tile.shape[1] / 8)
main_list = []
for i in range(len(CHARACTER_INFOS)):
    skins_list = []
    for j in range(8):
        skins_list.append(characters_tile[i * tile_height : (i+1) * tile_height, j * tile_width : (j+1) * tile_width].copy())
    main_list.append(skins_list)
# The following is not a mistake. The same closeups are used for both 2 and 3 players mode, but there was no clean way to point at the
#   first element of CHARACTER_CLOSEUPS both when players = 2 and players = 3, and then a different one when players >= 4. By adding it
#   two times, I can always get the correct set of closeups by calling CHARACTER_CLOSEUPS[players-2].
# Also, this should not lead in a waste of memory as .append() only adds a reference to the input, and does not copy it, so we're not
#   having two copies of the same massive list of closeups, but two small references to the same list of closeups
CHARACTER_CLOSEUPS.append(main_list)
CHARACTER_CLOSEUPS.append(main_list)

CHARACTER_ICONS = []
characters_tile = cv2.cvtColor(cv2.imread(os.path.join(res_path, "character_references", "icons.png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
tile_height = int(characters_tile.shape[0] / len(CHARACTER_INFOS))
tile_width = int(characters_tile.shape[1] / 8)
for i in range(len(CHARACTER_INFOS)):
    skins_list = []
    for j in range(8):
        skins_list.append(characters_tile[i * tile_height : (i+1) * tile_height, j * tile_width : (j+1) * tile_width].copy())
    CHARACTER_CLOSEUPS.append(skins_list)
    
EMPTY_ICONS = []
for i in range(MAX_PLAYERS):
    EMPTY_ICONS.append(cv2.imread(os.path.join(res_path, "character_references", "empty_icons", "P" + str(i+1) + ".png"), flags=cv2.IMREAD_UNCHANGED))
    

ANCHOR_POINT_LD = [24, 24, 22]
ANCHOR_POINT_STENCILS = []
for i in range(MAX_PLAYERS):
    ANCHOR_POINT_STENCILS.append(cv2.imread(os.path.join(res_path, "anchor_point", "P" + str(i+1) + ".png"), flags=cv2.IMREAD_UNCHANGED))


BACKGROUND_CLOSEUPS_BGRS = [[0, 4, 188],
                        [213, 102, 7],
                        [9, 172, 232]]


SMALL_DIGIT_WIDTH = 17
SMALL_DIGIT_HEIGHT = 21
SMALL_DIGIT_SEP = 19.5
SMALL_DIGIT_IMAGES = []
digits_tile = cv2.cvtColor(cv2.imread(os.path.join(res_path, "digits", "small_digits.png")), cv2.COLOR_BGR2GRAY)
for i in range(11):
    SMALL_DIGIT_IMAGES.append(digits_tile[:, i*SMALL_DIGIT_WIDTH : (i+1)*SMALL_DIGIT_WIDTH].copy())

BIG_DIGIT_WIDTH = 44
BIG_DIGIT_HEIGHT = 63
BIG_DIGIT_IMAGES = []
digits_tile = cv2.cvtColor(cv2.imread(os.path.join(res_path, "digits", "big_digits.png")), cv2.COLOR_BGR2GRAY)
for i in range(11):
    BIG_DIGIT_IMAGES.append(digits_tile[:, i*BIG_DIGIT_WIDTH : (i+1)*BIG_DIGIT_WIDTH].copy())


POLARIZATION_THRESHOLD = 40

# "Player pixel" = the pixel sampled used to determine the number of players based on its colour
PLAYER_PIXEL = (657, 972)

# PLAYERS_COLS[i] = The colour of the player pixel if there are [i+2] players
PLAYER_COLOURS = [numpy.array([0, 61, 150, 255]), 
                numpy.array([144, 114, 0, 255]), 
                numpy.array([0, 115, 20, 255])]

# LEFT_EDGE[i,j] = the y-position of the left edge of the rectangle containing the information
#   of the [j]th player when there are [i+2] players in total
LEFT_EDGE = [[175, 701],
            [19, 438, 857],
            [0, 320, 640, 960]]

# RIGHT_EDGE[i,j] = the y-position of the right edge of the rectangle containing the information
#   of the [j]th player when there are [i+2] players in total
RIGHT_EDGE = [[579, 1105],
            [423, 842, 1261],
            [320, 640, 960, 1280]]

PLACE_PIXEL_PY = 45
PLACE_PIXEL_RD = [-68, -68, -43]

PLACE_COLOURS = [numpy.array([255, 255, 0, 255]), 
                numpy.array([204, 204, 204, 255]), 
                numpy.array([241, 148, 17, 255]), 
                numpy.array([185, 171, 207, 255])]

NAME_WIDTH = 235
NAME_HEIGHT = 77
NAME_PY = [27, 27, 21]
NAME_LD = 0
WIN_OFFSET = 16

TIME_WIDTH = 191 + 40
TIME_HEIGHT = 92 + 40
TIME_PY = [404, 404, 399]
TIME_SEC_RD = [-90, -90, -81]
TIME_DEC_SEP = -48
TIME_MIN_SEP = -75

FALL_ICON_SIZE = [29, 29, 25]
FALL_ICON_YO = -43
FALL_ICON_LD = [37, 37, 31]
FALL_ICON_SEP = 33.3

GVN_DMG_YO = 105
TKN_DMG_YO = 175
DMG_RD = -58

SELFDESTR_YO = 35
SELFDESTR_RD = -21


#--- FREE STUFF ---#
del characters_tile
del tile_width
del tile_height
del main_list
del skins_list
del digits_tile