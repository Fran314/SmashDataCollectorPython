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
CHARACTER_NAMES = ["MARIO", "DONKEY KONG", "LINK", "SAMUS", "SAMUS OSCURA",
                    "YOSHI", "KIRBY", "FOX", "PIKACHU", "LUIGI", "NESS",
                    "CAPTAIN FALCON", "JIGGLYPUFF", "PEACH", "DAISY", "BOWSER",
                    "ICE CLIMBERS", "SHEIK", "ZELDA", "DR. MARIO", "PICHU",
                    "FALCO", "MARTH", "LUCINA", "LINK BAMBINO", "GANONDORF",
                    "MEWTWO", "ROY", "CHROM", "MR. GAME & WATCH", "META KNIGHT",
                    "PIT", "PIT OSCURO", "SAMUS TUTA ZERO", "WARIO", "SNAKE",
                    "IKE", "ALLENATORE DI POKÉMON", "DIDDY KONG", "LUCAS",
                    "SONIC", "KING DEDEDE", "OLIMAR", "LUCARIO", "R.O.B.",
                    "LINK CARTONE", "WOLF", "ABITANTE", "MEGA MAN",
                    "TRAINER DI WII FIT", "ROSALINDA E SFAVILLOTTO",
                    "LITTLE MAC", "GRENINJA", "GUERRIERO MII", "PALUTENA",
                    "PAC-MAN", "DARAEN", "SHULK", "BOWSER JUNIOR",
                    "DUO DUCK HUNT", "RYU", "KEN", "CLOUD", "CORRIN",
                    "BAYONETTA", "INKLING", "RIDLEY", "SIMON", "RICHTER",
                    "KING K. ROOL", "FUFFI", "INCINEROAR", "PIANTA PIRANHA",
                    "JOKER", "EROE", "BANJO E KAZOOIE", "TERRY", "BYLETH",
                    "MIN MIN", "STEVE"]

IMAGE_POLARIZATION_TRESHOLD = 40

FIRST_COL = numpy.array([251, 255, 0, 255])
SECOND_COL = numpy.array([204, 204, 204, 255])
THIRD_COL = numpy.array([240, 150, 12, 255])

# "Name rectangle" = the rectanlge containing the name of the caracter
NAME_WIDTH = 226 # Width of the name rectangle
NAME_HEIGHT = 77 # Height of the name rectangle

G1_NAME_X = 20 # X position of the name rectangle for G1
G2_NAME_X = 441 # X position of the name rectangle for G2
G3_NAME_X = 859 # X position of the name rectangle for G3
W_NAME_Y = 44 # Y position of the name rectangle for the winner
L_NAME_Y = 28 # Y position of the name rectangle for a loser

# "Time rectangle" = the rectangle containing the time the player survived
TIME_WIDTH = 200 # Width of the time rectangle
TIME_HEIGHT = 90 # Height of the time rectangle

TIME_Y = 391 # Y position of the time rectangle
G1_TIME_X = 198 # X position of the time rectangle for G1
G2_TIME_X = 614 # X position of the time rectangle for G2
G3_TIME_X = 1033 # X position of the time rectangle for G3
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
        character_name_errors[i] = optimalAlignError(name, CHARACTER_NAMES[i])
    
    return CHARACTER_NAMES[numpy.argmin(character_name_errors)]


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

#data_source = r'C:\Users\franc\Documents\Pyzo\SmashDataAnalyzer\data.jpg'
data_source = r'D:\Utente\Desktop\data.jpg'
data = cv2.imread(data_source, flags=cv2.IMREAD_UNCHANGED)
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)


#354, 44
G1_col = data[44,354]
G1_pos = numpy.argmin(numpy.array([numpy.linalg.norm(G1_col - FIRST_COL), numpy.linalg.norm(G1_col - SECOND_COL), numpy.linalg.norm(G1_col - THIRD_COL)]))+1

#773, 44
G2_col = data[44,773]
G2_pos = numpy.argmin(numpy.array([numpy.linalg.norm(G2_col - FIRST_COL), numpy.linalg.norm(G2_col - SECOND_COL), numpy.linalg.norm(G2_col - THIRD_COL)]))+1

#1191, 44
G3_col = data[44,1191]
G3_pos = numpy.argmin(numpy.array([numpy.linalg.norm(G3_col - FIRST_COL), numpy.linalg.norm(G3_col - SECOND_COL), numpy.linalg.norm(G3_col - THIRD_COL)]))+1

G1_character = getName(data, G1_NAME_X, G1_pos)
G2_character = getName(data, G2_NAME_X, G2_pos)
G3_character = getName(data, G3_NAME_X, G3_pos)

G1_time = getTimeRectangle(data, G1_TIME_X)
G2_time = getTimeRectangle(data, G2_TIME_X)
G3_time = getTimeRectangle(data, G3_TIME_X)

print("G1: %s [%s] - %s" %(G1_pos, normalizeTime(pytesseract.image_to_string(G1_time)), G1_character))
print("G2: %s [%s] - %s" %(G2_pos, normalizeTime(pytesseract.image_to_string(G2_time)), G2_character))
print("G3: %s [%s] - %s" %(G3_pos, normalizeTime(pytesseract.image_to_string(G3_time)), G3_character))

print("elapsed time: %.3f s" % (time.time() - t))