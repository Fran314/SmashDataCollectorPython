import numpy
from numpy import inf
import sys
import pytesseract
import cv2
import math


#--- INITIALIZATION ---#
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
#---   ---#


#--- CONSTANTS ---#
IMAGE_POLARIZATION_TRESHOLD = 40

FIRST_COL = numpy.array([251, 255, 0, 255])
SECOND_COL = numpy.array([204, 204, 204, 255])
THIRD_COL = numpy.array([240, 150, 12, 255])

# "Name rectangle" = the rectanlge containing the name of the caracter
NAME_WIDTH = 226 # Width of the name rectangle
NAME_HEIGHT = 77 # Height of the name rectangle

G1_NAME_X = 20 # X position of the name rectangle for G1
G2_NAME_X = 441 # X position of the name rectangle for G1
G3_NAME_X = 859 # X position of the name rectangle for G1
W_NAME_Y = 44 # Y position of the name rectangle for the winner
L_NAME_Y = 28 # Y position of the name rectangle for a loser
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


def getNameRectangle(data, pos_x, is_winner):
    if(is_winner == 1):
        name_rect = data[W_NAME_Y : (W_NAME_Y + NAME_HEIGHT), pos_x : (pos_x + NAME_WIDTH)]
    else:
        name_rect = data[L_NAME_Y : (L_NAME_Y + NAME_HEIGHT), pos_x : (pos_x + NAME_WIDTH)]

    return polarizeImage(name_rect)

#---   ---#


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

G1_character = getNameRectangle(data, G1_NAME_X, G1_pos)
G2_character = getNameRectangle(data, G2_NAME_X, G2_pos)
G3_character = getNameRectangle(data, G3_NAME_X, G3_pos)

cv2.imshow('G1', G1_character)
#cv2.waitKey(1000)
#cv2.destroyAllWindows()
cv2.imshow('G2', G2_character)
#cv2.waitKey(1000)
#cv2.destroyAllWindows()
cv2.imshow('G3', G3_character)
#cv2.waitKey(1000)
#cv2.destroyAllWindows()

print("G1: %s - %s" %(G1_pos, pytesseract.image_to_string(G1_character)))
print("G2: %s - %s" %(G2_pos, pytesseract.image_to_string(G2_character)))
print("G3: %s - %s" %(G3_pos, pytesseract.image_to_string(G3_character)))