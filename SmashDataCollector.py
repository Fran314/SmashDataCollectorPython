import time
import numpy
import sys
import os
import pytesseract
from cv2 import cv2
import re

#--- CUSTOMIZEABLE ---#
data_path = r'E:\Archivio\Programmazione\VSCode\SmashDataAnalyzer\data'
res_path = r'E:\Archivio\Programmazione\VSCode\SmashDataAnalyzer\res'
output_path = r'E:\Archivio\Programmazione\VSCode\SmashDataAnalyzer'

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

CHARACTER_NAMES = [["MARIO",                    "Mario"],
                    ["DONKEY KONG",             "Donkey Kong"],
                    ["LINK",                    "Link"],
                    ["SAMUS",                   "Samus"],
                    ["SAMUS OSCURA",            "Dark Samus"],
                    ["YOSHI",                   "Yoshi"],
                    ["KIRBY",                   "Kirby"],
                    ["FOX",                     "Fox"],
                    ["PIKACHU",                 "Pikachu"],
                    ["LUIGI",                   "Luigi"],
                    ["NESS",                    "Ness"],
                    ["CAPTAIN FALCON",          "Captain Falcon"],
                    ["JIGGLYPUFF",              "Jigglypuff"],
                    ["PEACH",                   "Peach"],
                    ["DAISY",                   "Daisy"],
                    ["BOWSER",                  "Bowser"],
                    ["ICE CLIMBERS",            "Ice Climbers"],
                    ["SHEIK",                   "Sheik"],
                    ["ZELDA",                   "Zelda"],
                    ["DR. MARIO",               "Dr. Mario"],
                    ["PICHU",                   "Pichu"],
                    ["FALCO",                   "Falco"],
                    ["MARTH",                   "Marth"],
                    ["LUCINA",                  "Lucina"],
                    ["LINK BAMBINO",            "Young Link"],
                    ["GANONDORF",               "Ganondorf"],
                    ["MEWTWO",                  "Mewtwo"],
                    ["ROY",                     "Roy"],
                    ["CHROM",                   "Chrom"],
                    ["MR. GAME & WATCH",        "Mr. Game & Watch"],
                    ["META KNIGHT",             "Meta Knight"],
                    ["PIT",                     "Pit"],
                    ["PIT OSCURO",              "Dark Pit"],
                    ["SAMUS TUTA ZERO",         "Zero Suit Samus"],
                    ["WARIO",                   "Wario"],
                    ["SNAKE",                   "Snake"],
                    ["IKE",                     "Ike"],
                    ["ALLENATORE DI POKÉMON",   "Pokémon Trainer"],
                    ["DIDDY KONG",              "Diddy Kong"],
                    ["LUCAS",                   "Lucas"],
                    ["SONIC",                   "Sonic"],
                    ["KING DEDEDE",             "King Dedede"],
                    ["OLIMAR",                  "Olimar"],
                    ["ALPH",                    "Olimar"],
                    ["LUCARIO",                 "Lucario"],
                    ["R.O.B.",                  "R.O.B."],
                    ["LINK CARTONE",            "Toon Link"],
                    ["WOLF",                    "Wolf"],
                    ["ABITANTE",                "Villager"],
                    ["MEGA MAN",                "Mega Man"],
                    ["TRAINER DI WII FIT",      "Wii Fit Trainer"],
                    ["ROSALINDA E SFAVILLOTTO", "Rosalina & Luma"],
                    ["LITTLE MAC",              "Little Mac"],
                    ["GRENINJA",                "Greninja"],
                    ["PALUTENA",                "Palutena"],
                    ["PAC-MAN",                 "Pac-Man"],
                    ["DARAEN",                  "Robin"],
                    ["SHULK",                   "Shulk"],
                    ["BOWSER JUNIOR",           "Bowser Jr."],
                    ["LARRY",                   "Bowser Jr."],
                    ["ROY",                     "Bowser Jr."],
                    ["WENDY",                   "Bowser Jr."],
                    ["IGGY",                    "Bowser Jr."],
                    ["MORTON",                  "Bowser Jr."],
                    ["LEMMY",                   "Bowser Jr."],
                    ["LUDWIG",                  "Bowser Jr."],
                    ["DUO DUCK HUNT",           "Duck Hunt"],
                    ["RYU",                     "Ryu"],
                    ["KEN",                     "Ken"],
                    ["CLOUD",                   "Cloud"],
                    ["CORRIN",                  "Corrin"],
                    ["BAYONETTA",               "Bayonetta"],
                    ["RAGAZZO INKLING",         "Inkling"],
                    ["RAGAZZA INKLING",         "Inkling"],
                    ["RIDLEY",                  "Ridley"],
                    ["SIMON",                   "Simon"],
                    ["RICHTER",                 "Richter"],
                    ["KING K. ROOL",            "King K. Rool"],
                    ["FUFFI",                   "Isabelle"],
                    ["INCINEROAR",              "Incineroar"],
                    ["PIANTA PIRANHA",          "Piranha Plant"],
                    ["JOKER",                   "Joker"],
                    ["EROE",                    "Hero"],
                    ["BANJO E KAZOOIE",         "Banjo & Kazooie"],
                    ["TERRY",                   "Terry"],
                    ["BYLETH",                  "Byleth"],
                    ["MIN MIN",                 "Min Min"],
                    ["STEVE",                   "Steve"],
                    ["ALEX",                    "Steve"],
                    ["ZOMBIE",                  "Steve"],
                    ["ENDERMAN",                "Steve"]]

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
        elif(c == ' ' or c =='-'):
            to_return += ' '
        elif(c == 'é'):
            to_return += 'e'
        elif(c == '&'):
            to_return += "and"
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


#--- PRELOAD MATCHES ---#
dirs = mergeSort(os.listdir(data_path))
if(len(dirs) % 2 != 0):
    print("Odd number of images present in data. Ignoring the last image.")
    dirs = dirs[:-1]
tot_matches = int(len(dirs)/2)

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


#--- ANALYZE MATCHES ---#
error_message = ""
output_strings = []
problematic_matches = []
for match_index in range(tot_matches):
    print(f'Match #{match_index+1} out of {tot_matches}')
    first_data = readImage(os.path.join(data_path, dirs[2*match_index]))
    second_data = readImage(os.path.join(data_path, dirs[2*match_index + 1]))

    try:
        #--- FIRST IMAGE ---#
        places = []
        characters = []
        times = []
        
        #--- GET NUMBER OF PLAYERS ---#
        # TODO: 2-3-4 players only
        player_col = first_data[PLAYER_PIXEL]
        players = numpy.argmin([numpy.linalg.norm(player_col - colour) for colour in PLAYER_COLS])+2

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
                character_name_ed.append(editDistance(curr_name, CHARACTER_NAMES[j][0]))
            min_index = numpy.argmin(character_name_ed)
            if(character_name_ed[min_index] >= len(CHARACTER_NAMES[min_index][0]) / 2):
                error_message = f'G{i+1}\'s name is too uncertain'
                raise InvalidData
            characters.append(CHARACTER_NAMES[min_index][1])

            #--- GET PLAYER TIME ---#
            curr_time = ""
            digit_x = RIGHT_EDGE[players-2][i] + TIME_SEC_RD[players-2]
            digit_y = TIME_PY[players-2]
            digit_image = submat(first_data, digit_y, digit_x, BIG_DIGIT_HEIGHT, BIG_DIGIT_WIDTH)
            digit_image = polarizeImage(digit_image, POLARIZATION_THRESHOLD)
            digit = getClosestDigit(digit_image, BIG_DIGIT_IMAGES)
            if(digit != -1):
                curr_time = str(digit)

                digit_x += TIME_DEC_SEP
                digit_image = submat(first_data, digit_y, digit_x, BIG_DIGIT_HEIGHT, BIG_DIGIT_WIDTH)
                digit_image = polarizeImage(digit_image, POLARIZATION_THRESHOLD)
                digit = getClosestDigit(digit_image, BIG_DIGIT_IMAGES)
                curr_time = str(digit) + curr_time

                digit_x += TIME_MIN_SEP
                digit_image = submat(first_data, digit_y, digit_x, BIG_DIGIT_HEIGHT, BIG_DIGIT_WIDTH)
                digit_image = polarizeImage(digit_image, POLARIZATION_THRESHOLD)
                digit = getClosestDigit(digit_image, BIG_DIGIT_IMAGES)
                curr_time = str(digit) + ":" + curr_time
            times.append(curr_time)

            #--- PLAYER - ERROR CHECKING ---#
            if(places[i] < 1 or places[i] > players):
                error_message = f'G{i+1}\'s position ({places[i]}) is invalid'
                raise InvalidData
            if(isValidTime(times[i]) == False):
                error_message = f'G{i+1}\'s time ({times[i]}) is invalid'
                raise InvalidData
        
        #--- FIRST DATA - ERROR CHECKING ---#
        if(isValidFirstData(places, times) == False):
            error_message = f'positions ({places}) and times ({times}) are invalid'
            #raise InvalidData


        #--- SECOND IMAGE ---#
        falls = []
        given_damages = []
        taken_damages = []
        for i in range(players):
            #--- ESTABLISH ANCHOR POINTS ---#
            stencil_height = ap_stencils[i].shape[0]
            stencil_width = ap_stencils[i].shape[1]
            data_height = second_data.shape[0]
            min_norm = imageDistance(ap_stencils[i], submat(second_data, 0, LEFT_EDGE[players-2][i] + ANCHOR_POINT_LD[players-2], stencil_height, stencil_width))
            pos_min = 0
            for j in range(data_height - stencil_height):
                curr_norm = imageDistance(ap_stencils[i], submat(second_data, j, LEFT_EDGE[players-2][i] + ANCHOR_POINT_LD[players-2], stencil_height, stencil_width))
                if(curr_norm < min_norm):
                    min_norm = curr_norm
                    pos_min = j
            anchor_point = pos_min

            #--- GET PLAYER FALLS ---#
            curr_fall_list = []
            fall_icon_px = LEFT_EDGE[players-2][i] + FALL_ICON_LD[players-2]
            fall_icon_py = anchor_point + FALL_ICON_YO
            for j in range(LIVES):
                fall_image = submat(second_data, fall_icon_py, int(fall_icon_px), FALL_ICON_SIZE[players-2], FALL_ICON_SIZE[players-2])
                fall_image = cv2.resize(fall_image, (50,50))
                killer = getClosestPlayer(fall_image, null_images[i], characters)
                if(killer == -1):
                    break
                if(killer == i):
                    error_message = f'G{i+1}\'s first killer read as itself'
                    raise InvalidData
                curr_fall_list.append(killer + 1)
                fall_icon_px += FALL_ICON_SEP

            #--- GET PLAYER SELFDESTRUCTS ---#
            digit_y = anchor_point + SELFDESTR_YO
            digit_x = RIGHT_EDGE[players-2][i] + SELFDESTR_RD
            digit_image = submat(second_data, digit_y, digit_x, SMALL_DIGIT_HEIGHT, SMALL_DIGIT_WIDTH)
            digit_image = polarizeImage(digit_image, POLARIZATION_THRESHOLD)
            curr_selfdestruct = getClosestDigit(digit_image, SMALL_DIGIT_IMAGES)
            for j in range(curr_selfdestruct):
                curr_fall_list.append(i+1)

            falls.append(curr_fall_list)

            #--- FALLS AND SELFDESTRUCTS - ERROR CHECKING ---#
            if(places[i] != 1 and len(curr_fall_list) != LIVES):
                error_message = f'G{i+1} (not in first place) died a number of times different from {LIVES} ({curr_fall_list})'
                raise InvalidData
            if(places[i] == 1 and len(curr_fall_list) >= LIVES):
                error_message = f'G{i+1} (in first place) died more than {LIVES-1} times ({curr_fall_list})'
                raise InvalidData


            #--- GET PLAYER GIVEN DAMAGE ---#
            given_damage = 0
            digit_y = anchor_point + GVN_DMG_YO
            digit_x = RIGHT_EDGE[players-2][i] + DMG_RD
            digit = 0
            decimal_place = 1
            while(digit != -1):
                digit_image = submat(second_data, digit_y, int(digit_x), SMALL_DIGIT_HEIGHT, SMALL_DIGIT_WIDTH)
                digit_image = polarizeImage(digit_image, POLARIZATION_THRESHOLD)
                digit = getClosestDigit(digit_image, SMALL_DIGIT_IMAGES)
                if(digit != -1):
                    given_damage += digit*decimal_place
                digit_x -= SMALL_DIGIT_SEP
                decimal_place *= 10
            given_damages.append(given_damage)
            
            #--- GET PLAYER TAKEN DAMAGE ---#
            taken_damage = 0
            digit_y = anchor_point + TKN_DMG_YO
            digit_x = RIGHT_EDGE[players-2][i] + DMG_RD
            digit = 0
            decimal_place = 1
            while(digit != -1):
                digit_image = submat(second_data, digit_y, int(digit_x), SMALL_DIGIT_HEIGHT, SMALL_DIGIT_WIDTH)
                digit_image = polarizeImage(digit_image, POLARIZATION_THRESHOLD)
                digit = getClosestDigit(digit_image, SMALL_DIGIT_IMAGES)
                if(digit != -1):
                    taken_damage += digit*decimal_place
                digit_x -= SMALL_DIGIT_SEP
                decimal_place *= 10
            taken_damages.append(taken_damage)
        
        #--- SECOND DATA - ERROR CHECKING ---#
        taken_given_dmg_difference = 0
        for i in range(players):
            taken_given_dmg_difference += taken_damages[i] - given_damages[i]
        if(taken_given_dmg_difference < 0 or taken_given_dmg_difference >= TAKEN_GIVEN_DMG_THRESHOLD):
            error_message = f'too big of a difference between total taken damage and total given damage ({taken_given_dmg_difference})'
            raise InvalidData


        #--- CONVERT MATCH DATA TO A STRING ---#
        output_strings.append(convertMatchToString(dirs[2*match_index], players, characters, places, times, falls, given_damages, taken_damages))

    except InvalidData:
        problematic_matches.append([match_index, error_message])
        print(f'Problem: {error_message}')
        output_strings.append("")


#--- HALFWAY CHECKUP ---#
if(len(problematic_matches) > 0):
    print(f'elapsed time: {(time.time() - t):.3f} s')

    
#--- WRITE OUTPUT ---#
output_file = open(os.path.join(output_path, "output.tsv"), 'w')
for match in output_strings[:-1]:
    output_file.write(match)
    output_file.write("\n")
output_file.write(output_strings[-1])
output_file.close()


#--- HANDLE PROBLEMATIC MATCHES ---#
if(len(problematic_matches) > 0):
    print(f'Unable to read data of {len(problematic_matches)} matches out of {tot_matches}. Please enter the data manually.')
match_counter = 0
try:
    for problematic_match in problematic_matches:
        print(f'Problematic match #{match_counter+1} of {len(problematic_matches)} (match #{problematic_match[0]+1})')
        print(f'Problem: {problematic_match[1]}')

        first_data = readImage(os.path.join(data_path, dirs[2*problematic_match[0]]))
        second_data = readImage(os.path.join(data_path, dirs[2*problematic_match[0] + 1]))
        valid_data = False
        while(valid_data == False):
            valid_data = True
            print('At any time, you can enter "SKIP" to skip the current match or enter "SKIP ALL" to skip all the remaining matches.')
            try:
                #--- FIRST IMAGE ---#
                cv2.destroyAllWindows()
                showImage(cv2.resize(first_data, (640, 360)), f'match {problematic_match[0]+1}/{int(len(dirs)/2)}, first screenshot', 20)
                characters = []
                positions = []
                times = []
                for i in range(PLAYERS):
                    #--- GET PLAYER CHARACTER ---#
                    regex = f'({")|(".join([name[0] for name in CHARACTER_NAMES])})'
                    player_character = readInput(f'Enter G{i+1} character: ', regex)
                    for name in CHARACTER_NAMES:
                        if(name[0] == player_character.upper()):
                            player_character = name[1]
                            break
                    characters.append(player_character)

                    #--- GET PLAYER POSITION ---#
                    regex = f'[1-{PLAYERS}]'
                    player_position = readInput(f'Enter G{i+1} position: ', regex)
                    positions.append(int(player_position))

                    #--- GET PLAYER TIME ---#
                    regex = "([0-9]:[0-5][0-9])?"
                    player_time = readInput(f'Enter G{i+1} time (m:ss): ', regex)
                    times.append(player_time)
                
                #--- FIRST DATA - ERROR CHECKING ---#
                if(isValidFirstData(positions, times) == False):
                    error_message = f'positions ({positions}) and times ({times}) are invalid'
                    raise InvalidData


                #--- SECOND IMAGE ---#
                cv2.destroyAllWindows()
                showImage(cv2.resize(second_data, (640, 360)), f'match {problematic_match[0]+1}/{int(len(dirs)/2)}, second screenshot', 20)
                falls = []
                given_damages = []
                taken_damages = []
                for i in range(PLAYERS):
                    #--- GET PLAYER FALLS ---#
                    regex = "([" + "".join(map(str, range(1, i+1))) + "".join(map(str, range(i+2, PLAYERS+1))) + "](,[" + "".join(map(str, range(1, i+1))) + "".join(map(str, range(i+2, PLAYERS+1))) + "]){0," + str(LIVES - 1) + "})?"
                    player_falls_string = readInput(f'Enter G{i+1} falls (separated by commas): ', regex)
                    fall_list = []
                    for j in range(len(player_falls_string)):
                        if(j % 2 == 0):
                            fall_list.append(int(player_falls_string[j]))
                    falls.append(fall_list)

                    #--- GET PLAYER SELFDESTRUCTS --#
                    regex = f'[0-{LIVES}]'
                    player_selfdestruct_string = readInput(f'Enter G{i+1} selfdestructs: ', regex)
                    for j in range(int(player_selfdestruct_string)):
                        fall_list.append(i+1)

                    #--- ERROR CHECKING ---#
                    if(positions[i] != 1 and len(fall_list) != LIVES):
                        error_message = f'G{i+1} (not in first position) died a number of times different from {LIVES} ({fall_list})'
                        raise InvalidData
                    if(positions[i] == 1 and len(fall_list) >= LIVES):
                        error_message = f'G{i+1} (in first position) died more than {LIVES-1} times ({fall_list})'
                        raise InvalidData
                    
                    #--- GET PLAYER GIVEN DAMAGE ---#
                    regex = "[0-9]+"
                    player_given_damage = readInput(f'Enter G{i+1} given damage (without percent): ', regex)
                    given_damages.append(int(player_given_damage))

                    #--- GET PLAYER TAKEN DAMAGE ---#
                    regex = "[0-9]+"
                    player_taken_damage = readInput(f'Enter G{i+1} taken damage (without percent): ', regex)
                    taken_damages.append(int(player_taken_damage))
                    
                #--- SECOND DATA - ERROR CHECKING ---#
                taken_given_dmg_difference = 0
                for i in range(PLAYERS):
                    taken_given_dmg_difference += taken_damages[i] - given_damages[i]
                if(taken_given_dmg_difference < 0 or taken_given_dmg_difference >= TAKEN_GIVEN_DMG_THRESHOLD):
                    error_message = f'too big of a difference between total taken damage and total given damage ({taken_given_dmg_difference})'
                    raise InvalidData


                #--- CONVERT MATCH DATA TO A STRING ---#
                output_strings[problematic_match[0]] = convertMatchToString(dirs[2*problematic_match[0]], PLAYERS, characters, positions, times, falls, given_damages, taken_damages)

            except InvalidData:
                print("The data inserted has an error or is self-contradictory in some way.")
                print("Problem: " + error_message)
                print("Please check and insert the data again. If the data doesn't have any error and isn't actually self-contradicotry, please skip and add the data manually to the output.")
                valid_data = False
            except Skip:
                valid_data = True
        
        cv2.destroyAllWindows()
        print("The data inserted is valid!")
        match_counter += 1
except SkipAll:
    pass

#--- WRITE OUTPUT ---#
output_file = open(os.path.join(output_path, "output.tsv"), 'w')
for match in output_strings[:-1]:
    output_file.write(match)
    output_file.write("\n")
output_file.write(output_strings[-1])
output_file.close()


#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
input("Press Enter to continue...")