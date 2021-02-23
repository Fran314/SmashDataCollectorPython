import os
import numpy
import sys
from cv2 import cv2
from customizable import RES_PATH
from functions import readTSV, addBackground


MAX_PLAYERS = 4 # Max number of players


CHARACTER_INFOS = readTSV(os.path.join(RES_PATH, "character_references", "characters_info.tsv"))
CHARACTER_CLOSEUPS = []
# 2 and 3 players closeups
characters_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "character_references", "23-closeups.png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
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
# 4 players closeups
characters_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "character_references", "4-closeups.png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
tile_height = int(characters_tile.shape[0] / len(CHARACTER_INFOS))
tile_width = int(characters_tile.shape[1] / 8)
main_list = []
for i in range(len(CHARACTER_INFOS)):
    skins_list = []
    for j in range(8):
        skins_list.append(characters_tile[i * tile_height : (i+1) * tile_height, j * tile_width : (j+1) * tile_width].copy())
    main_list.append(skins_list)
CHARACTER_CLOSEUPS.append(main_list)

CHARACTER_CLOSEUPS_MASKS = [[(cv2.imread(os.path.join(RES_PATH, "character_references", "23-lose_mask.png")) / 255).astype(numpy.uint8),
                            (cv2.imread(os.path.join(RES_PATH, "character_references", "23-win_mask.png")) / 255).astype(numpy.uint8)],
                            [(cv2.imread(os.path.join(RES_PATH, "character_references", "23-lose_mask.png")) / 255).astype(numpy.uint8),
                            (cv2.imread(os.path.join(RES_PATH, "character_references", "23-win_mask.png")) / 255).astype(numpy.uint8)],
                            [(cv2.imread(os.path.join(RES_PATH, "character_references", "4-lose_mask.png")) / 255).astype(numpy.uint8),
                            (cv2.imread(os.path.join(RES_PATH, "character_references", "4-win_mask.png")) / 255).astype(numpy.uint8)]]

CHARACTER_ICONS = []
characters_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "character_references", "icons.png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
tile_height = int(characters_tile.shape[0] / len(CHARACTER_INFOS))
tile_width = int(characters_tile.shape[1] / 8)
for i in range(len(CHARACTER_INFOS)):
    skins_list = []
    for j in range(8):
        skins_list.append(characters_tile[i * tile_height : (i+1) * tile_height, j * tile_width : (j+1) * tile_width].copy())
    CHARACTER_ICONS.append(skins_list)


BACKGROUND_CLOSEUPS_BGRS = [[140, 144, 144],
                        [0, 4, 188],
                        [213, 102, 7],
                        [9, 172, 232],
                        [49, 153, 15]]

BACKGROUND_ICONS_BGRS = [[114, 114, 114],
                        [32, 27, 172],
                        [181, 90, 21],
                        [0, 125, 166],
                        [22, 131, 22]]

DARK_BAND_BGRS = [[39, 39, 39],
                    [12, 13, 64],
                    [65, 34, 13],
                    [0, 55, 70],
                    [32, 70, 22]]

ANCHOR_POINT_STENCILS = []
for i in range(MAX_PLAYERS + 1):
    stencil = numpy.zeros((66, 6, 3), dtype=numpy.uint8)
    stencil[0 : 26, :] = numpy.array(DARK_BAND_BGRS[i], dtype=numpy.uint8)
    stencil[26 : 66, :] = numpy.array(BACKGROUND_ICONS_BGRS[i], dtype=numpy.uint8)
    ANCHOR_POINT_STENCILS.append(stencil)

AP_STENCIL_WIDTH = ANCHOR_POINT_STENCILS[0].shape[1]
AP_STENCIL_HEIGHT = ANCHOR_POINT_STENCILS[0].shape[0]
AP_STENCILS_RD = -13 - AP_STENCIL_WIDTH
AP_STENCILS_TOPY = [172, 172, 157]
AP_STENCILS_BOTY = [443, 443, 434]
AP_SPAN_RANGE = int(AP_STENCIL_HEIGHT * 3.0 / 4.0)

DARK_BAND_SEP = 74

SMALL_DIGIT_WIDTH = 17
SMALL_DIGIT_HEIGHT = 21
SMALL_DIGIT_SEP = 19.5
SMALL_DIGIT_IMAGES = []
MIN_SMALL_DIGIT_IMAGES = []
MAX_SMALL_DIGIT_IMAGES = []
digits_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "digits", "small_digits.png")), cv2.COLOR_BGR2GRAY)
for i in range(11):
    SMALL_DIGIT_IMAGES.append(digits_tile[:, i*SMALL_DIGIT_WIDTH : (i+1)*SMALL_DIGIT_WIDTH].copy())
digits_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "digits", "min_small_digits.png")), cv2.COLOR_BGR2GRAY)
for i in range(10):
    MIN_SMALL_DIGIT_IMAGES.append(digits_tile[:, i*SMALL_DIGIT_WIDTH : (i+1)*SMALL_DIGIT_WIDTH].copy())
digits_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "digits", "max_small_digits.png")), cv2.COLOR_BGR2GRAY)
for i in range(10):
    MAX_SMALL_DIGIT_IMAGES.append(digits_tile[:, i*SMALL_DIGIT_WIDTH : (i+1)*SMALL_DIGIT_WIDTH].copy())

BIG_DIGIT_WIDTH = 44
BIG_DIGIT_HEIGHT = 63
BIG_DIGIT_IMAGES = []
MIN_BIG_DIGIT_IMAGES = []
MAX_BIG_DIGIT_IMAGES = []
digits_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "digits", "big_digits.png")), cv2.COLOR_BGR2GRAY)
for i in range(12):
    BIG_DIGIT_IMAGES.append(digits_tile[:, i*BIG_DIGIT_WIDTH : (i+1)*BIG_DIGIT_WIDTH].copy())
digits_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "digits", "min_big_digits.png")), cv2.COLOR_BGR2GRAY)
for i in range(10):
    MIN_BIG_DIGIT_IMAGES.append(digits_tile[:, i*BIG_DIGIT_WIDTH : (i+1)*BIG_DIGIT_WIDTH].copy())
digits_tile = cv2.cvtColor(cv2.imread(os.path.join(RES_PATH, "digits", "max_big_digits.png")), cv2.COLOR_BGR2GRAY)
for i in range(10):
    MAX_BIG_DIGIT_IMAGES.append(digits_tile[:, i*BIG_DIGIT_WIDTH : (i+1)*BIG_DIGIT_WIDTH].copy())


POLARIZATION_THRESHOLD = 40

# "Player pixels" = the pixels sampled used to determine the number of players
PLAYER_PIXELS = [[(409, 179), (409, 705)],
                [(409, 23), (409, 442), (409, 861)],
                [(397, 4), (397, 324), (397, 644), (397, 964)]]

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

PLACE_COLOURS = [numpy.array([0, 255, 255]), 
                numpy.array([204, 204, 204]), 
                numpy.array([17, 148, 241]), 
                numpy.array([207, 171, 185])]

PLAYER_TYPE_COLOURS = [numpy.array([146, 146, 146]),
                        numpy.array([51, 51, 211]),
                        numpy.array([237, 106, 39]),
                        numpy.array([0, 170, 228]),
                        numpy.array([41, 153, 12])]


TIME_WIDTH = 191 + 40
TIME_HEIGHT = 92 + 40
TIME_PY = [404, 404, 399]
TIME_SEC_RD = [-90, -90, -81]
TIME_DEC_SEP = -48
TIME_MIN_SEP = -75

FALL_ICON_SIZE = [29, 29, 25]
FALL_ICON_YO = [32, 32, 33]
FALL_ICON_LD = [37, 37, 31]
FALL_ICON_SEP = 33.3

GVN_DMG_YO = 180
TKN_DMG_YO = 250
SELFDESTR_YO = 110

# LANGUAGE DEPENDENT
DMG_RD = [-67, -58, -67, -67]
SELFDESTR_RD = [-67, -37, -37, -67]

CHARACTER_CLOSEUP_HEIGHT = [255, 255, 245]


LANGUAGE_LD = 10
LANGUAGE_YO = [425, 425, 418]
LANGUAGE_WIDTH = [120, 120, 105]
LANGUAGE_HEIGHT = 45

languages_tile23 = cv2.imread(os.path.join(RES_PATH, "languages", "23-languages.png"))
languages_tile4 = cv2.imread(os.path.join(RES_PATH, "languages", "4-languages.png"))
LANGUAGES = [[[languages_tile23[0 : 45, : ], 0],    # Japanese
            [languages_tile23[45 : 90, : ], 1],     # English
            [languages_tile23[90 : 135, : ], 2],    # French
            [languages_tile23[135 : 180, : ], 2],   # Spanish
            [languages_tile23[180 : 225, : ], 2],   # German
            [languages_tile23[225 : 270, : ], 1],   # Italian
            [languages_tile23[270 : 315, : ], 1],   # Dutch
            [languages_tile23[315 : 360, : ], 1],   # Russian
            [languages_tile23[360 : 405, : ], 3],   # Chinese (traditional)
            [languages_tile23[405 : 450, : ], 3],   # Chinese (simplified)
            [languages_tile23[450 : 495, : ], 0]],  # Korean

            [[languages_tile23[0 : 45, : ], 0],     # Japanese
            [languages_tile23[45 : 90, : ], 1],     # English
            [languages_tile23[90 : 135, : ], 2],    # French
            [languages_tile23[135 : 180, : ], 2],   # Spanish
            [languages_tile23[180 : 225, : ], 2],   # German
            [languages_tile23[225 : 270, : ], 1],   # Italian
            [languages_tile23[270 : 315, : ], 1],   # Dutch
            [languages_tile23[315 : 360, : ], 1],   # Russian
            [languages_tile23[360 : 405, : ], 3],   # Chinese (traditional)
            [languages_tile23[405 : 450, : ], 3],   # Chinese (simplified)
            [languages_tile23[450 : 495, : ], 0]],  # Korean
            
            [[languages_tile4[0 : 45, : ], 0],      # Japanese
            [languages_tile4[45 : 90, : ], 1],      # English
            [languages_tile4[90 : 135, : ], 2],     # French
            [languages_tile4[135 : 180, : ], 2],    # Spanish
            [languages_tile4[180 : 225, : ], 2],    # German
            [languages_tile4[225 : 270, : ], 1],    # Italian
            [languages_tile4[270 : 315, : ], 1],    # Dutch
            [languages_tile4[315 : 360, : ], 1],    # Russian
            [languages_tile4[360 : 405, : ], 3],    # Chinese (traditional)
            [languages_tile4[405 : 450, : ], 3],    # Chinese (simplified)
            [languages_tile4[450 : 495, : ], 0]]]   # Korean


#--- FREE STUFF ---#
del characters_tile
del tile_width
del tile_height
del main_list
del skins_list
del digits_tile