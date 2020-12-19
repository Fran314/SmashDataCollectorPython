import time
import numpy
import sys
import os
import pytesseract
from cv2 import cv2
import re

#--- CUSTOMIZEABLE ---#
data_path = r'D:\Utente\Desktop\smash\CharactersReferences'
res_path = r'E:\Archivio\Programmazione\VSCode\SmashDataAnalyzer\res'
output_path = r'E:\Archivio\Programmazione\VSCode\SmashDataAnalyzer'

character_path = r'D:\Utente\Desktop\smash\faces'

LIVES = 3
TAKEN_GIVEN_DMG_THRESHOLD = 30


#--- INITIALIZATION ---#
t = time.time()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


#--- CLASSES ---#
class InvalidData(Exception):
    pass

class Skip(Exception):
    pass

class SkipAll(Exception):
    pass


#--- CONSTANTS ---#
MAX_PLAYERS = 4 # Max number of players
PLAYERS = 3 # Number of players

SMALL_DIGIT_IMAGES = []
BIG_DIGIT_IMAGES = []

POLARIZATION_THRESHOLD = 40

# "Player pixel" = the pixel sampled used to determine the number of players based on its colour
PLAYER_PIXEL = (657, 972)

# PLAYERS_COLS[i] = The colour of the player pixel if there are [i+2] players
PLAYER_COLS = [numpy.array([0, 61, 150, 255]), 
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

ANCHOR_POINT_LD = [24, 24, 22]

PLACE_PIXEL_PY = 45
PLACE_PIXEL_RD = [-68, -68, -43]

PLACE_COLS = [numpy.array([255, 255, 0, 255]), 
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

BIG_DIGIT_WIDTH = 44
BIG_DIGIT_HEIGHT = 63
SMALL_DIGIT_WIDTH = 17
SMALL_DIGIT_HEIGHT = 21
SMALL_DIGIT_SEP = 19.5

GVN_DMG_YO = 105
TKN_DMG_YO = 175
DMG_RD = -58

SELFDESTR_YO = 35
SELFDESTR_RD = -21


#--- FUNCTIONS ---#
def editDistance(arg0, arg1):
    curr_buffer = numpy.array(range(len(arg1) + 1))
    for i in range(len(arg0)):
        prev_buffer = numpy.copy(curr_buffer)
        curr_buffer = numpy.zeros(len(arg1) + 1)
        curr_buffer[0] = i+1
        for j in range(len(arg1)):
            pij = 0 if(arg0[i] == arg1[j]) else 1
            curr_buffer[j+1] = min(prev_buffer[j] + pij, curr_buffer[j] + 1, prev_buffer[j+1] + 1)

    return curr_buffer[len(arg1)]


def mergeSort(arg):
    if(len(arg) <= 1):
        return arg
    
    mid = int(len(arg)/2)
    left = mergeSort(arg[:mid])
    right = mergeSort(arg[mid:])

    k = 0
    i = 0
    j = 0
    while(k < len(arg) and i < len(left) and j < len(right)):
        if(left[i] < right[j]):
            arg[k] = left[i]
            i += 1
        else:
            arg[k] = right[j]
            j += 1
        k += 1
    
    while(i < len(left)):
        arg[k] = left[i]
        i += 1
        k += 1
    while(j < len(right)):
        arg[k] = right[j]
        j += 1
        k += 1

    return arg


def imageDistance(stencil, image):
    # stencil_shape = stencil.shape
    # image_shape = image.shape
    # if(stencil_shape != image_shape):
    #     if(len(stencil_shape) == 1):
    #         stencil_norm = numpy.linalg.norm(stencil)
    #     else:
    #         stencil_norm = numpy.linalg.norm(stencil[:,:])
        
    #     if(len(image_shape) == 1):
    #         image_norm = numpy.linalg.norm(image)
    #     else:
    #         image_norm = numpy.linalg.norm(image[:,:])

    #     return 3*numpy.max(2*stencil_norm, 2*image_norm)

    distance = numpy.linalg.norm(numpy.multiply((stencil[:,:,0]/1.0 - image[:,:,0]/1.0), stencil[:,:,3]/255.0))
    distance += numpy.linalg.norm(numpy.multiply((stencil[:,:,1]/1.0 - image[:,:,1]/1.0), stencil[:,:,3]/255.0))
    distance += numpy.linalg.norm(numpy.multiply((stencil[:,:,2]/1.0 - image[:,:,2]/1.0), stencil[:,:,3]/255.0))

    return distance


def polarizeImage(image_to_polarize, treshold):
    height = image_to_polarize.shape[0]
    width = image_to_polarize.shape[1]

    to_return = numpy.copy(image_to_polarize)

    for i in range(height):
        for j in range(width):
            if(numpy.linalg.norm(image_to_polarize[i,j] - numpy.array([255, 255, 255, 255])) > treshold):
                to_return[i,j] = numpy.array([0, 0, 0, 255])

    return to_return


def submat(matrix, i, j, di, dj):
    return matrix[i : i + di, j : j + dj]


def getClosestPlayer(image, null_image, characters):
    distances = [imageDistance(null_image, image)]
    for i in range(len(characters)):
        character_distances = []
        for j in range(8):
            path = os.path.join(res_path, "icons", folderizeName(characters[i]), str(j+1) + ".png")
            character_image = readImage(path)
            distance = imageDistance(character_image, image)
            character_distances.append(distance)
        distances.append(numpy.min(character_distances))
    return numpy.argmin(distances) - 1


def getClosestDigit(image, digit_images):
    distances = []
    for i in range(11):
        distances.append(imageDistance(digit_images[i], image))
    
    digit = numpy.argmin(distances)
    return digit-1


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


def normalizeDamage(arg):
    to_return = ""
    for c in arg:
        if((ord(c) >= 48 and ord(c) <= 57)):
            to_return += c
    
    return to_return


def folderizeName(arg):
    to_return = ""
    for c in arg:
        if(ord(c) >= 97 and ord(c) <= 122):
            to_return += c
        elif(ord(c) >= 65 and ord(c) <= 90):
            to_return += chr(ord(c) + 32)
    return to_return


def isValidTime(arg):
    if(arg == ""):
        return True
    if(len(arg) != 4):
        return False
    
    if(arg[1] != ':'):
        return False

    if(ord(arg[0]) < 48 or ord(arg[0]) > 57
        or ord(arg[2]) < 48 or ord(arg[2]) > 53
        or ord(arg[3]) < 48 or ord(arg[3]) > 57):
        return False

    return True


def isValidFirstData(positions, times):
    if(len(positions) != len(times)):
        return False
    
    if(len(times) < 2):
        return False

    for i in range(len(times)):
        if(positions[i] == 1 and times[i] != ""):
            return False
        if(positions[i] != 1 and times[i] == ""):
            return False

    order = []
    for i in range(len(times)):
        pos = len(order)
        while(pos > 0 and positions[i] < positions[order[pos-1]]):
            pos -= 1
        order.insert(pos, i)

    if(positions[order[0]] != 1 or positions[order[1]] != 2):
        return False

    for i in range(2, len(times)):
        if(positions[order[i]] == i+1 and time2int(times[order[i]]) >= time2int(times[order[i-1]])):
            return False
        if(positions[order[i]] != i+1 and time2int(times[order[i]]) != time2int(times[order[i-1]])):
            return False
    
    return True


def time2int(arg):
    return 60 * int(arg[0]) + int(arg[2:4])
def time2string(arg):
    if(arg == ""):
        return ""
    else:
        return str(time2int(arg))


def convertMatchToString(file_name, players, characters, positions, times, falls, given_damages, taken_damages):
    match_string = file_name[6:8] + "/" + file_name[4:6] + "/" + file_name[0:4] + "\t"
    match_string += str(players) + "\t"
    for i in range(players):
        match_string += characters[i] + "\t"
        match_string += str(positions[i]) + "\t"
        match_string += ','.join(map(str, falls[i])) + "\t"
        match_string += str(given_damages[i]) + "\t"
        match_string += str(taken_damages[i]) + "\t"
        match_string += time2string(times[i])
        if(i < players - 1):
            match_string += "\t"
    
    return match_string


def readInput(prompt, regex):
    valid_input = False
    while(valid_input == False):
        user_input = input(prompt)
        if(user_input == "SKIP"):
            raise Skip
        elif(user_input == "SKIP ALL"):
            raise SkipAll
        elif(re.fullmatch(regex, user_input.upper())):
            valid_input = True
        else:
            print("Invalid input")
    return user_input


def readImage(path):
    image = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    return image
def showImage(image, text = "image", delay = 0):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    cv2.namedWindow(text)
    cv2.moveWindow(text, 20, 20)
    cv2.imshow(text, image)
    cv2.waitKey(delay)

def readList(path):
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
            if(c != ','):
                curr_val += c
            else:
                values.append(curr_val)
                curr_val = ""
        if(curr_val != ""):
            values.append(curr_val)
        to_return.append(values)
    
    return to_return
            


#--- PRELOAD MATCHES ---#
matches = mergeSort(os.listdir(data_path))

#--- LOAD RES ---#
ap_stencils = []
null_images = []
for i in range(MAX_PLAYERS):
    ap_stencil_path = res_path + r'\ap_stencils\G' + str(i+1) + r'_ap_stencil.png'
    image = readImage(ap_stencil_path)
    ap_stencils.append(image)

    null_image_path = res_path + r'\null_images\G' + str(i+1) + r'_null_image.png'
    image = readImage(null_image_path)
    null_images.append(image)

for i in range(11):
    digit_path = os.path.join(res_path, "small_digits",  str(i-1) + ".png")
    SMALL_DIGIT_IMAGES.append(readImage(digit_path))
    digit_path = os.path.join(res_path, "big_digits",  str(i-1) + ".png")
    BIG_DIGIT_IMAGES.append(readImage(digit_path))

CHARACTER_NAMES = readList(r'D:\Utente\Desktop\character_info.csv')

#--- ANALYZE MATCHES ---#
for match in matches:
    first_data = readImage(os.path.join(data_path, match))

    #--- FIRST IMAGE ---#
    places = []
    characters = []
    
    players = 3

    for i in range(players):
        #--- GET PLAYER PLACE ---#
        curr_place_col = first_data[PLACE_PIXEL_PY, RIGHT_EDGE[players-2][i] + PLACE_PIXEL_RD[players-2]]
        curr_place = numpy.argmin([numpy.linalg.norm(curr_place_col - place_col) for place_col in PLACE_COLS])+1
        places.append(curr_place)

        #--- GET PLAYER CHARACTER ---#
        if(places[i] != 1):
            curr_name_rect = submat(first_data, NAME_PY[players-2], LEFT_EDGE[players-2][i] + NAME_LD, NAME_HEIGHT, NAME_WIDTH)
        else:
            curr_name_rect = submat(first_data, NAME_PY[players-2] + WIN_OFFSET, LEFT_EDGE[players-2][i] + NAME_LD, NAME_HEIGHT, NAME_WIDTH)
        curr_name_rect = polarizeImage(curr_name_rect, POLARIZATION_THRESHOLD)
        curr_name = normalizeName(pytesseract.image_to_string(curr_name_rect))
        character_name_ed = []
        for j in range(len(CHARACTER_NAMES)):
            character_name_ed.append(editDistance(curr_name, CHARACTER_NAMES[j][0].upper()))
        min_index = numpy.argmin(character_name_ed)
        name = ""
        character_image = first_data[153 : 153 + 240, LEFT_EDGE[players-2][i] : RIGHT_EDGE[players-2][i]]
        if(character_name_ed[min_index] >= len(CHARACTER_NAMES[min_index][0]) / 3):
            print(f'G{i+1}\'s name is too uncertain')

            showImage(character_image, "boh", 20)

            regex = f'({")|(".join([name[0].upper() for name in CHARACTER_NAMES])})'
            player_name = readInput(f'Enter G{i+1} character: ', regex)
            for name in CHARACTER_NAMES:
                if(name[0].upper() == player_name.upper()):
                    player_name = name[2]
                    break
        else:
            player_name = CHARACTER_NAMES[min_index][2]
            
        folder_path = os.path.join(character_path, player_name)
        try:
            os.mkdir(folder_path)
        except Exception:
            pass
        already_existing = len(os.listdir(folder_path))
        character_image = cv2.cvtColor(character_image, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(folder_path, str(already_existing) + ".png"), character_image)

#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')