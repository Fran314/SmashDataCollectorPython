import time
import numpy
import sys
import pytesseract
from cv2 import cv2

#--- INITIALIZATION ---#
t = time.time()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
#---   ---#


#--- CONSTANTS ---#
CHARACTER_NAMES = [["MARIO",					"MARIO"],
                    ["DONKEY KONG",				"DONKEY KONG"],
                    ["LINK",					"LINK"],
                    ["SAMUS",					"SAMUS"],
                    ["SAMUS OSCURA",			"DARK SAMUS"],
                    ["YOSHI",					"YOSHI"],
                    ["KIRBY",					"KIRBY"],
                    ["FOX",						"FOX"],
                    ["PIKACHU",					"PIKACHU"],
                    ["LUIGI",					"LUIGI"],
                    ["NESS",					"NESS"],
                    ["CAPTAIN FALCON",			"CAPTAIN FALCON"],
                    ["JIGGLYPUFF",				"JIGGLYPUFF"],
                    ["PEACH",					"PEACH"],
                    ["DAISY",					"DAISY"],
                    ["BOWSER",					"BOWSER"],
                    ["ICE CLIMBERS",			"ICE CLIMBERS"],
                    ["SHEIK",					"SHEIK"],
                    ["ZELDA",					"ZELDA"],
                    ["DR. MARIO",				"DR. MARIO"],
                    ["PICHU",					"PICHU"],
                    ["FALCO",					"FALCO"],
                    ["MARTH",					"MARTH"],
                    ["LUCINA",					"LUCINA"],
                    ["LINK BAMBINO",			"YOUNG LINK"],
                    ["GANONDORF",				"GANONDORF"],
                    ["MEWTWO",					"MEWTWO"],
                    ["ROY",						"ROY"],
                    ["CHROM",					"CHROM"],
                    ["MR. GAME & WATCH",		"MR. GAME & WATCH"],
                    ["META KNIGHT",				"META KNIGHT"],
                    ["PIT",						"PIT"],
                    ["PIT OSCURO",				"DARK PIT"],
                    ["SAMUS TUTA ZERO",			"ZERO SUIT SAMUS"],
                    ["WARIO",					"WARIO"],
                    ["SNAKE",					"SNAKE"],
                    ["IKE",						"IKE"],
                    ["ALLENATORE DI POKÉMON",	"POKÉMON TRAINER"],
                    ["DIDDY KONG",				"DIDDY KONG"],
                    ["LUCAS",					"LUCAS"],
                    ["SONIC",					"SONIC"],
                    ["KING DEDEDE",				"KING DEDEDE"],
                    ["OLIMAR",					"OLIMAR"],
                    ["LUCARIO",					"LUCARIO"],
                    ["R.O.B.",					"R.O.B."],
                    ["LINK CARTONE",			"TOON LINK"],
                    ["WOLF",					"WOLF"],
                    ["ABITANTE",				"VILLAGER"],
                    ["MEGA MAN",				"MEGA MAN"],
                    ["TRAINER DI WII FIT",		"WII FIT TRAINER"],
                    ["ROSALINDA E SFAVILLOTTO",	"ROSALINA & LUMA"],
                    ["LITTLE MAC",				"LITTLE MAC"],
                    ["GRENINJA",				"GRENINJA"],
                    ["GUERRIERO MII",			"MII FIGHTER"],
                    ["LOTTATORE MII",			"MII FIGHTER"],
                    ["SPADACCINO MII",			"MII FIGHTER"],
                    ["FUCILIERE MII",			"MII FIGHTER"],
                    ["PALUTENA",				"PALUTENA"],
                    ["PAC-MAN",					"PAC-MAN"],
                    ["DARAEN",					"ROBIN"],
                    ["SHULK",					"SHULK"],
                    ["BOWSER JUNIOR",			"BOWSER JR."],
                    ["LARRY",					"BOWSER JR."],
                    ["ROY",						"BOWSER JR."],
                    ["WENDY",					"BOWSER JR."],
                    ["IGGY",					"BOWSER JR."],
                    ["MORTON",					"BOWSER JR."],
                    ["LEMMY",					"BOWSER JR."],
                    ["LUDWIG",					"BOWSER JR."],
                    ["DUO DUCK HUNT",			"DUCK HUNT"],
                    ["RYU",						"RYU"],
                    ["KEN",						"KEN"],
                    ["CLOUD",					"CLOUD"],
                    ["CORRIN",					"CORRIN"],
                    ["BAYONETTA",				"BAYONETTA"],
                    ["RAGAZZO INKLING",			"INKLING"],
                    ["RAGAZZA INKLING",			"INKLING"],
                    ["RIDLEY",					"RIDLEY"],
                    ["SIMON",					"SIMON"],
                    ["RICHTER",					"RICHTER"],
                    ["KING K. ROOL",			"KING K. ROOL"],
                    ["FUFFI",					"ISABELLE"],
                    ["INCINEROAR",				"INCINEROAR"],
                    ["PIANTA PIRANHA",			"PIRANHA PLANT"],
                    ["JOKER",					"JOKER"],
                    ["EROE",					"HERO"],
                    ["BANJO E KAZOOIE",			"BANJO & KAZOOIE"],
                    ["TERRY",					"TERRY"],
                    ["BYLETH",					"BLYETH"],
                    ["MIN MIN",					"MIN MIN"],
                    ["STEVE",					"STEVE"],
                    ["ALEX",					"STEVE"],
                    ["ZOMBIE",					"STEVE"],
                    ["ENDERMAN",				"STEVE"]]

IMAGE_POLARIZATION_TRESHOLD = 40

PLAYERS = 3 # Number of players

FIRST_COL = numpy.array([251, 255, 0, 255]) # Color of the first place #1 icon
SECOND_COL = numpy.array([204, 204, 204, 255]) # Color of the second place #2 icon
THIRD_COL = numpy.array([240, 150, 12, 255]) # Color of the third place #3 icon

# "Sample point" = the pixel sampled to determine the position of each player based
#   on the colour of the pixel (see previous constants)
SAMPLE_POINT_Y = 44 # Y position of the sample point for every player
SAMPLE_POINT_Xs = [354, # X position of the sample point for G1
                    773, # X position of the sample point for G2
                    1191] # X position of the sample point for G3

# "Name rectangle" = the rectanlge containing the name of the caracter
NAME_WIDTH = 226 # Width of the name rectangle
NAME_HEIGHT = 77 # Height of the name rectangle

NAME_Xs = [20,      # X position of the name rectangle for G1
            441,    # X position of the name rectangle for G2
            859]    # X position of the name rectangle for G3
W_NAME_Y = 44 # Y position of the name rectangle for the winner
L_NAME_Y = 28 # Y position of the name rectangle for a loser

# "Time rectangle" = the rectangle containing the time the player survived
TIME_WIDTH = 200 # Width of the time rectangle
TIME_HEIGHT = 90 # Height of the time rectangle

TIME_Y = 391 # Y position of the time rectangle
TIME_Xs = [198,     # X position of the time rectangle for G1
            614,    # X position of the time rectangle for G2
            1033]   # X position of the time rectangle for G3
#---   ---#


#--- DEFINITIONS ---#
def polarizeImage(image_to_polarize):
    height = image_to_polarize.shape[0]
    width = image_to_polarize.shape[1]

    for i in range(height):
        for j in range(width):
            if(numpy.linalg.norm(image_to_polarize[i,j] - numpy.array([255, 255, 255, 255])) > IMAGE_POLARIZATION_TRESHOLD):
                image_to_polarize[i,j] = numpy.array([0, 0, 0, 255])

    return image_to_polarize


def getName(data, pos_x, is_winner):
    if(is_winner == 1):
        name_rect = data[W_NAME_Y : (W_NAME_Y + NAME_HEIGHT), pos_x : (pos_x + NAME_WIDTH)]
    else:
        name_rect = data[L_NAME_Y : (L_NAME_Y + NAME_HEIGHT), pos_x : (pos_x + NAME_WIDTH)]

    name_rect = polarizeImage(name_rect)
    name = normalizeName(pytesseract.image_to_string(name_rect))
    character_name_errors = numpy.zeros(len(CHARACTER_NAMES), dtype=int)
    for i in range(len(CHARACTER_NAMES)):
        character_name_errors[i] = optimalAlignError(name, CHARACTER_NAMES[i][0])
    
    return CHARACTER_NAMES[numpy.argmin(character_name_errors)][1]


def getTimeRectangle(data, pos_x):
    return cv2.resize(polarizeImage(data[TIME_Y : (TIME_Y + TIME_HEIGHT), pos_x : (pos_x + TIME_WIDTH)]), (2*TIME_WIDTH, 2*TIME_HEIGHT))


def normalizeName(arg):
    to_return = ""
    for c in arg:
        if((ord(c) >= 65 and ord(c) <= 90) or (ord(c) == 32)):
            to_return += c
        elif(ord(c) >= 97 and ord(c) <= 122):
            to_return += chr(ord(c) - 32)
    
    return to_return


def normalizeTime(arg):
    to_return = ""
    for c in arg:
        if((ord(c) >= 48 and ord(c) <= 58)):
            to_return += c
    
    return to_return


def optimalAlignError(arg0, arg1):
    curr_buffer = numpy.array(range(len(arg1) + 1))
    for i in range(len(arg0)):
        brev_buffer = numpy.copy(curr_buffer)
        curr_buffer = numpy.zeros(len(arg1) + 1, dtype=int)
        curr_buffer[0] = i+1
        for j in range(len(arg1)):
            pij = 0 if(arg0[i] == arg1[j]) else 1
            curr_buffer[j+1] = min(brev_buffer[j] + pij, curr_buffer[j] + 1, brev_buffer[j+1] + 1)

    return curr_buffer[len(arg1)]
#---   ---#


data_source = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\data_first.jpg'
#data_source = r'D:\Utente\Desktop\data.jpg'
data = cv2.imread(data_source, flags=cv2.IMREAD_UNCHANGED)
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)

positions = []
for i in range(PLAYERS):
    col = data[SAMPLE_POINT_Y, SAMPLE_POINT_Xs[i]]
    positions.append(numpy.argmin(numpy.array([numpy.linalg.norm(col - FIRST_COL), numpy.linalg.norm(col - SECOND_COL), numpy.linalg.norm(col - THIRD_COL)]))+1)

characters = []
for i in range(PLAYERS):
    characters.append(getName(data, NAME_Xs[i], positions[i]))

times = []
for i in range(PLAYERS):
    time_rect = getTimeRectangle(data, TIME_Xs[i])
    times.append(normalizeTime(pytesseract.image_to_string(time_rect)))

for i in range(PLAYERS):
    print(f'G{i+1}: {positions[i]} [{times[i]}] - {characters[i]}')


#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
#--- ---#