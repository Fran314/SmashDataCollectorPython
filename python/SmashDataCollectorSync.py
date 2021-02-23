import time
import numpy
import sys
import os
from cv2 import cv2
import re
import functions as fun
import resources as res
import customizable as custom

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

def polarizeImage(image_to_polarize, treshold):
    height = image_to_polarize.shape[0]
    width = image_to_polarize.shape[1]
    to_return = numpy.copy(image_to_polarize)
    for i in range(height):
        for j in range(width):
            if(numpy.linalg.norm(image_to_polarize[i,j] - numpy.array([255, 255, 255])) > treshold):
                to_return[i,j] = numpy.array([0, 0, 0])
    return cv2.cvtColor(to_return, cv2.COLOR_BGR2GRAY)

def submat(matrix, i, j, di, dj):
    return matrix[i : i + di, j : j + dj].copy()

def getClosestCharacterByCloseup(image, character_closeups, mask, background_bgr):
    distances = []
    for i in range(len(character_closeups)):
        character_distances = []
        for j in range(8):
            curr_closeup = addBackground(character_closeups[i][j], background_bgr)
            character_distances.append(numpy.sum(numpy.multiply(cv2.absdiff(curr_closeup, image), mask)))
        distances.append(numpy.min(character_distances))
    return numpy.argmin(distances)

def getClosestPlayerByIcon(image, icons, background_bgr):
    empty_icon = numpy.full_like(image, background_bgr[0])
    empty_icon[:,:,1] = numpy.full((image.shape[0], image.shape[1]), background_bgr[1])
    empty_icon[:,:,2] = numpy.full((image.shape[0], image.shape[1]), background_bgr[2])
    distances = [numpy.sum(cv2.absdiff(empty_icon, image))]
    for i in range(len(icons)):
        character_distances = []
        for j in range(len(icons[i])):
            curr_icon = addBackground(icons[i][j], background_bgr)
            character_distances.append(numpy.sum(cv2.absdiff(curr_icon, image)))
        distances.append(numpy.min(character_distances))
    return numpy.argmin(distances) - 1

def getClosestDigit(image, digit_images):
    distances = []
    for i in range(len(digit_images)):
        distances.append(numpy.sum(cv2.absdiff(digit_images[i], image)))
    digit = numpy.argmin(distances)
    return digit-1

def getMinMaxDigit(image, min_digits, max_digits):
    has_min = [(min <= image).all() for min in min_digits]
    is_in_max = [(image <= max).all() for max in max_digits]
    valid = -1
    for i in range(10):
        if(has_min[i] and is_in_max[i]):
            if(valid == -1):
                valid = i
            else:
                return -1
    return valid

def getClosestLanguageType(image, languages):
    distances = []
    for i in range(len(languages)):
        distances.append(numpy.sum(cv2.absdiff(languages[i][0], image)))
    index = numpy.argmin(distances)
    return languages[index][1]

def addBackground(image_bgra, colour_bgr):
    to_return = numpy.zeros((image_bgra.shape[0], image_bgra.shape[1], 3))
    mask = image_bgra[:, :, 3] * (1 / 255.0)
    antimask = 1 - mask
    to_return[:,:, 0] = numpy.multiply(image_bgra[:, :, 0], mask) + colour_bgr[0] * antimask
    to_return[:,:, 1] = numpy.multiply(image_bgra[:, :, 1], mask) + colour_bgr[1] * antimask
    to_return[:,:, 2] = numpy.multiply(image_bgra[:, :, 2], mask) + colour_bgr[2] * antimask
    return to_return.astype(numpy.uint8)

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
    if(len(arg) == 4):
        return 60 * int(arg[0]) + int(arg[2:4])
    else:
        return 60 * int(arg[0:2]) + int(arg[3:5])
def time2string(arg):
    if(arg == ""):
        return ""
    else:
        return str(time2int(arg))

def convertMatchToString(file_name, characters, positions, times, falls, given_damages, taken_damages):
    players = len(characters)
    match_string = file_name[6:8] + "/" + file_name[4:6] + "/" + file_name[0:4] + "\t"
    match_string += str(players) + "\t"
    for i in range(players):
        match_string += characters[i] + "\t"
        match_string += str(positions[i]) + "\t"
        match_string += time2string(times[i]) + "\t"
        match_string += ','.join(map(str, falls[i])) + "\t"
        match_string += str(given_damages[i]) + "\t"
        match_string += str(taken_damages[i])
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
            if(c == '#'):
                if(curr_val != ""):
                    values.append(curr_val)
                    curr_val = ""
                break
            elif(c == '\t'):
                if(curr_val != ""):
                    values.append(curr_val)
                    curr_val = ""
            else:
                curr_val += c
        if(curr_val != ""):
            values.append(curr_val)
        if(len(values) > 0):
            to_return.append(values)
    return to_return

def showImage(image, text = "image", delay = 0):
    cv2.namedWindow(text)
    cv2.moveWindow(text, 20, 20)
    cv2.imshow(text, image)
    cv2.waitKey(delay)


class Match():
    def __init__(self, file_name, first_scrn, second_scrn):
        self.date = file_name[6:8] + "/" + file_name[4:6] + "/" + file_name[0:4]
        self.first_scrn = first_scrn
        self.second_scrn = second_scrn

    def init_data(self):
        #--- FIRST SCREENSHOT DATA ---#
        self.types = [-1 for i in range(self.players_amount)]
        self.positions = [-1 for i in range(self.players_amount)]
        self.characters = [-1 for i in range(self.players_amount)]
        self.times = ["" for i in range(self.players_amount)]

        #--- SECOND SCREENSHOT DATA ---#
        self.anchor_points = [-1 for i in range(self.players_amount)]
        self.falls = ["" for i in range(self.players_amount)]
        self.given_damages = [-1 for i in range(self.players_amount)]
        self.taken_damages = [-1 for i in range(self.players_amount)]


    def inputPlayersAmount(self):
        # print(f'Unable to tell how many players were in this match. Please, solve this issue manually.')
        # cv2.destroyAllWindows()
        # showImage(cv2.resize(self.first_scrn, (640, 360)), f'match #{self.match_index}, first screenshot', 20)
        regex = f'[2-{res.MAX_PLAYERS}]'
        self.players_amount = int(readInput(f'Enter number of players: ', regex))
    def getPlayersAmount(self):
        self.players_amount = -1
        for i in range(3):
            if(all([numpy.linalg.norm(self.first_scrn[pixel]) < 20 for pixel in res.PLAYER_PIXELS[i]])):
                self.players_amount = i+2
                break
        if(self.players_amount == -1):
            print(f'Unable to tell how many players were in this match. Please, solve this issue manually.')
            return -1
        else:
            return 0

    def getPlayerType(self, player_index):
        type_identifier_pixel = (res.TIME_PY[self.players_amount-2], res.RIGHT_EDGE[self.players_amount-2][player_index] + int(res.TIME_SEC_RD[self.players_amount-2] / 2))
        type_identifier_colour = self.first_scrn[type_identifier_pixel]
        player_type = numpy.argmin([numpy.linalg.norm(type_identifier_colour - colour) for colour in res.PLAYER_TYPE_COLOURS])
        self.types[player_index] = player_type
        return 0

    def getPlayerPosition(self, player_index):
        position_identifier_pixel = (res.PLACE_PIXEL_PY, res.RIGHT_EDGE[self.players_amount-2][player_index] + res.PLACE_PIXEL_RD[self.players_amount-2])
        position_identifier_colour = self.first_scrn[position_identifier_pixel]
        player_position = numpy.argmin([numpy.linalg.norm(position_identifier_colour - colour) for colour in res.PLACE_COLOURS])+1
        self.positions[player_index] = player_position
        return 0

    def getPlayerCharacter(self, player_index):
        character_closeup = self.first_scrn[138 : 138 + res.CHARACTER_CLOSEUP_HEIGHT[self.players_amount-2], res.LEFT_EDGE[self.players_amount-2][player_index] : res.RIGHT_EDGE[self.players_amount-2][player_index]].copy()
        player_character = getClosestCharacterByCloseup(character_closeup, res.CHARACTER_CLOSEUPS[self.players_amount-2], res.CHARACTER_CLOSEUPS_MASKS[self.players_amount-2][1 if curr_place == 1 else 0], res.BACKGROUND_CLOSEUPS_BGRS[self.types[player_index]])
        self.characters[player_index] = player_character
        return 0

    def inputPlayerTime(self, player_index):
        regex = "([1-9]?[0-9]:[0-5][0-9])?"
        self.times[player_index] = readInput(f'Enter P{player_index+1} time (m:ss): ', regex)
    def getPlayerTime(self, player_index):
        player_time = ""
        digit_px = [ res.RIGHT_EDGE[self.players_amount-2][player_index] + res.TIME_SEC_RD[self.players_amount-2] ]
        digit_px.append(digit_px[0] + res.TIME_DEC_SEP)
        digit_px.append(digit_px[1] + res.TIME_MIN_SEP)
        digit_px.append(digit_px[2] + res.TIME_DEC_SEP)
        digit_py = res.TIME_PY[self.players_amount-2]
        for i in range(4):
            digit_image = self.first_scrn[digit_py : digit_py + res.BIG_DIGIT_HEIGHT, digit_px[i] : digit_px[i] + res.BIG_DIGIT_WIDTH]
            digit_image = polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            digit = getMinMaxDigit(digit_image, res.MIN_BIG_DIGIT_IMAGES, res.MAX_BIG_DIGIT_IMAGES)
            if(digit == -1):
                break
            player_time = str(digit) + player_time

        if(player_time == ""):
            self.times[player_index] = player_time
            return 0
        regex = "([1-9]?[0-9]:[0-5][0-9])?"
        elif(len(player_time) >= and re.fullmatch(regex, player_time[:-2] + ':' + player_time[-2:])):
            self.times[player_index] = player_time[:-2] + ':' + player_time[-2:]
            return 0
        print(f'Unable to read P{player_index+1}\'s time. Please, solve this issue manually.')
        return -1

    def getAnchorPoints(self, player_index):
        ap_area_py = res.AP_STENCILS_TOPY[self.players_amount - 2]
        ap_area_dy = res.AP_STENCILS_BOTY[self.players_amount - 2] - ap_py
        ap_area_px = res.RIGHT_EDGE[self.players_amount-2][player_index] + res.AP_STENCILS_RD
        ap_area_dx = res.AP_STENCIL_WIDTH
        ap_area = submat(self.second_scrn, ap_area_py, ap_area_px, ap_area_dy, ap_area_dx)
        ap_matches = cv2.matchTemplate(ap_area, res.ANCHOR_POINT_STENCILS[types[player_index]], cv2.TM_CCOEFF)
        anchor_point = -1
        for j in range(ap_matches.shape[0]):
            upper_span = min(j, res.AP_SPAN_RANGE)
            lower_span = min(ap_matches.shape[0] - j - 1, res.AP_SPAN_RANGE)
            region = ap_matches[j - upper_span : j + lower_span]
            if(numpy.argmax(region) == upper_span):
                left_sample_py = 
                if(numpy.sum(fun.polarizeImage(fun.submat(second_data, res.AP_STENCILS_TOPY[self.players_amount - 2] + j, res.RIGHT_EDGE[self.players_amount-2][player_index] + res.SELFDESTR_RD[language], res.AP_STENCIL_HEIGHT, res.SMALL_DIGIT_WIDTH), 170)) == 0 and
                    numpy.sum(fun.polarizeImage(fun.submat(second_data, res.AP_STENCILS_TOPY[self.players_amount - 2] + j + res.DARK_BAND_SEP, res.RIGHT_EDGE[self.players_amount-2][player_index] + res.SELFDESTR_RD[language], res.AP_STENCIL_HEIGHT, res.SMALL_DIGIT_WIDTH), 170)) > 0):
                    anchor_point = res.AP_STENCILS_TOPY[self.players_amount - 2] + j
        
        if(anchor_point == -1):
            print(f'Unable to find P{player_index+1}\'s anchor point. Please, solve this issue manually.')
            return -1
        
        self.anchor_points[player_index] = anchor_point
        return 0

    def inputPlayerFalls(self, player_index):
        regex = "([" + "".join(map(str, range(1, curr_player_index+1))) + "".join(map(str, range(curr_player_index+2, players_amount+1))) + "](,[" + "".join(map(str, range(1, curr_player_index+1))) + "".join(map(str, range(curr_player_index+2, players_amount+1))) + "]){0," + str(custom.LIVES - 1) + "})?"
        player_falls_string = fun.readInput(f'Enter P{curr_player_index+1} falls (separated by commas): ', regex)
        player_falls = []
        for j in range(len(player_falls_string)):
            if(j % 2 == 0):
                fall_list.append(int(player_falls_string[j]))
    def getPlayerFalls(self, player_index):
        player_falls = []
        fall_icon_px = res.LEFT_EDGE[self.players_amount-2][player_index] + res.FALL_ICON_LD[self.players_amount-2]
        fall_icon_py = self.anchor_points[player_index] + res.FALL_ICON_YO[self.players_amount-2]
        for j in range(custom.LIVES):
            fall_image = submat(self.second_scrn, fall_icon_py, int(fall_icon_px), res.FALL_ICON_SIZE[self.players_amount-2], res.FALL_ICON_SIZE[self.players_amount-2])
            fall_image = cv2.resize(fall_image, (50,50))
            killer = getClosestPlayerByIcon(fall_image, [res.CHARACTER_ICONS[index] for index in self.characters], res.BACKGROUND_ICONS_BGRS[self.types[player_index]])
            if(killer == -1):
                break
            if(killer == player_index):
                # fun.showImage(cv2.resize(self.second_scrn, (640, 360)), f'match #{match_index}, second screenshot', 20)
                # print(f'Couldn\'t read P{player_index+1}\'s falls properly. Please, add it manually.')
                # break
                return -1
            player_falls.append(killer + 1)
            fall_icon_px += res.FALL_ICON_SEP

        digit_py = anchor_point + res.SELFDESTR_YO
        digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD[language]
        digit_image = fun.submat(second_data, digit_py, digit_x, res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
        digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
        curr_selfdestruct = fun.getMinMaxDigit(digit_image, res.MIN_SMALL_DIGIT_IMAGES, res.MAX_SMALL_DIGIT_IMAGES)
        if(curr_selfdestruct == -1):
            fun.showImage(cv2.resize(second_data, (640, 360)), f'match #{match_index}, second screenshot', 20)
            print(f'Couldn\'t read P{curr_player_index+1}\'s selfdestructions properly. Please, add it manually.')
            
            regex = f'[0-{custom.LIVES}]'
            curr_selfdestruct = int(fun.readInput(f'Enter P{curr_player_index+1} selfdestructions: ', regex))
            
            cv2.destroyAllWindows()

        for j in range(curr_selfdestruct):
            curr_fall_list.append(curr_player_index+1)

        

#--- INITIALIZATION ---#
t = time.time()

if(custom.LIVES > 8):
    print("The number of lives can't be greater than 8.")
    input("Press Enter to continue...")
    sys.exit(0)

#--- FIND MATCHES ---#
dirs = fun.mergeSort(os.listdir(custom.DATA_PATH))
if(len(dirs) % 2 != 0):
    print("Odd number of images present in data. Ignoring the last image.")
    dirs = dirs[:-1]
tot_matches = int(len(dirs)/2)

#--- ANALYZE MATCHES ---#
error_message = ""
output_strings = []
problematic_matches = []
for match_index in range(tot_matches):
    print(f'Match #{match_index+1} out of {tot_matches}')
    first_data = cv2.imread(os.path.join(custom.DATA_PATH, dirs[2*match_index]))
    second_data = cv2.imread(os.path.join(custom.DATA_PATH, dirs[2*match_index+1]))

    #--- FIRST IMAGE ---#
    types = []
    positions = []
    character_indices = []
    times = []
    
    #--- GET NUMBER OF PLAYERS ---#
    players_amount = 0
    for i in range(3):
        if(all([numpy.linalg.norm(first_data[pixel]) < 20 for pixel in res.PLAYER_PIXELS[i]])):
            players_amount = i+2
            break
    if(players_amount == 0):
        fun.showImage(cv2.resize(first_data, (640, 360)), f'match #{match_index}, first screenshot', 20)
        print(f'Unable to tell how many players were in this match. Please, add it manually.')
        regex = f'[2-{res.MAX_PLAYERS}]'
        players_amount = int(fun.readInput(f'Enter number of players: ', regex))
        cv2.destroyAllWindows()
    
    language_image = fun.submat(first_data, res.LANGUAGE_YO[players_amount-2], res.LEFT_EDGE[players_amount-2][0] + res.LANGUAGE_LD, res.LANGUAGE_HEIGHT, res.LANGUAGE_WIDTH[players_amount-2])
    language = fun.getClosestLanguageType(language_image, res.LANGUAGES[players_amount-2])


    for curr_player_index in range(players_amount):
        #--- GET PLAYER TYPE ---#
        type_colour = first_data[res.TIME_PY[players_amount-2], res.RIGHT_EDGE[players_amount-2][curr_player_index] + int(res.TIME_SEC_RD[players_amount-2] / 2)]
        curr_player_type = numpy.argmin([numpy.linalg.norm(type_colour - colour) for colour in res.PLAYER_TYPE_COLOURS])
        types.append(curr_player_type)

        #--- GET PLAYER PLACE ---#
        place_colour = first_data[res.PLACE_PIXEL_PY, res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.PLACE_PIXEL_RD[players_amount-2]]
        curr_position = numpy.argmin([numpy.linalg.norm(place_colour - colour) for colour in res.PLACE_COLOURS])+1
        positions.append(curr_position)

        #--- GET PLAYER CHARACTER ---#
        character_closeup = first_data[138 : 138 + res.CHARACTER_CLOSEUP_HEIGHT[players_amount-2], res.LEFT_EDGE[players_amount-2][curr_player_index] : res.RIGHT_EDGE[players_amount-2][curr_player_index]].copy()
        curr_character_index = fun.getClosestCharacterByCloseup(character_closeup, res.CHARACTER_CLOSEUPS[players_amount-2], res.CHARACTER_CLOSEUPS_MASKS[players_amount-2][1 if curr_place == 1 else 0], res.BACKGROUND_CLOSEUPS_BGRS[curr_player_type])
        character_indices.append(curr_character_index)

        #--- GET PLAYER TIME ---#
        curr_time = ""
        digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.TIME_SEC_RD[players_amount-2]
        digit_y = res.TIME_PY[players_amount-2]
        digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
        digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
        digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
        if(digit == 10):
            fun.showImage(cv2.resize(first_data, (640, 360)), f'match #{match_index}, first screenshot', 20)
            print(f'Unable to read P{curr_player_index+1}\'s time. Please, add it manually.')
            regex = "([1-9]?[0-9]:[0-5][0-9])?"
            curr_time = fun.readInput(f'Enter P{curr_player_index+1} time (m:ss): ', regex)
            cv2.destroyAllWindows()
        elif(digit != -1):
            curr_time = str(digit)

            digit_x += res.TIME_DEC_SEP
            digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
            curr_time = str(digit) + curr_time

            digit_x += res.TIME_MIN_SEP
            digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
            curr_time = str(digit) + ":" + curr_time

            digit_x += res.TIME_DEC_SEP
            digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
            if(digit == -1):
                fun.showImage(cv2.resize(first_data, (640, 360)), f'match #{match_index}, first screenshot', 20)
                print(f'Unable to read P{curr_player_index+1}\'s time. Please, add it manually.')
                regex = "([1-9]?[0-9]:[0-5][0-9])?"
                curr_time = fun.readInput(f'Enter P{curr_player_index+1} time (m:ss): ', regex)
                cv2.destroyAllWindows()
            elif(digit != 10):
                curr_time = str(digit) + curr_time
        times.append(curr_time)
    
    #--- FIRST DATA - ERROR CHECKING ---#
    if(fun.isValidFirstData(positions, times) == False):
        fun.showImage(cv2.resize(first_data, (640, 360)), f'match #{match_index}, first screenshot', 20)
        print(f'The positions and times read don\'t make sense. Please, add them manually.')
        positions = []
        times = []
        for curr_player_index in range(players_amount):
            regex = f'[1-{players_amount}]'
            player_position = fun.readInput(f'Enter P{curr_player_index+1} position: ', regex)
            positions.append(int(player_position))

            regex = "([1-9]?[0-9]:[0-5][0-9])?"
            player_time = fun.readInput(f'Enter P{curr_player_index+1} time (m:ss): ', regex)
            times.append(player_time)
        cv2.destroyAllWindows()


    #--- SECOND IMAGE ---#
    falls = []
    given_damages = []
    taken_damages = []
    for curr_player_index in range(players_amount):
        #--- ESTABLISH ANCHOR POINTS ---#
        ap_area = second_data[res.AP_STENCILS_TOPY[players_amount - 2] : res.AP_STENCILS_BOTY[players_amount - 2], 
                            res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.AP_STENCILS_RD : res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.AP_STENCILS_RD + res.AP_STENCIL_WIDTH].copy()
        ap_matches = cv2.matchTemplate(ap_area, res.ANCHOR_POINT_STENCILS[types[curr_player_index]], cv2.TM_CCOEFF)
        anchor_point = 0
        for j in range(ap_matches.shape[0]):
            upper_span = min(j, res.AP_SPAN_RANGE)
            lower_span = min(ap_matches.shape[0] - j - 1, res.AP_SPAN_RANGE)
            region = ap_matches[j - upper_span : j + lower_span]
            if(numpy.argmax(region) == upper_span):
                if(numpy.sum(fun.polarizeImage(fun.submat(second_data, res.AP_STENCILS_TOPY[players_amount - 2] + j, res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD[language], res.AP_STENCIL_HEIGHT, res.SMALL_DIGIT_WIDTH), 170)) == 0 and
                    numpy.sum(fun.polarizeImage(fun.submat(second_data, res.AP_STENCILS_TOPY[players_amount - 2] + j + res.DARK_BAND_SEP, res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD[language], res.AP_STENCIL_HEIGHT, res.SMALL_DIGIT_WIDTH), 170)) > 0):
                    anchor_point = res.AP_STENCILS_TOPY[players_amount - 2] + j
        if(anchor_point == 0):
            fun.showImage(cv2.resize(second_data, (640, 360)), f'match #{match_index}, second screenshot', 20)
            print(f'Couldn\'t find anchor point for P{curr_player_index+1}. Please, add their data manually.')
            
            regex = "([" + "".join(map(str, range(1, curr_player_index+1))) + "".join(map(str, range(curr_player_index+2, players_amount+1))) + "](,[" + "".join(map(str, range(1, curr_player_index+1))) + "".join(map(str, range(curr_player_index+2, players_amount+1))) + "]){0," + str(custom.LIVES - 1) + "})?"
            player_falls_string = fun.readInput(f'Enter P{curr_player_index+1} falls (separated by commas): ', regex)
            curr_fall_list = []
            for j in range(len(player_falls_string)):
                if(j % 2 == 0):
                    curr_fall_list.append(int(player_falls_string[j]))

            regex = f'[0-{custom.LIVES}]'
            player_selfdestruct_string = fun.readInput(f'Enter P{curr_player_index+1} selfdestructs: ', regex)
            for j in range(int(player_selfdestruct_string)):
                curr_fall_list.append(curr_player_index+1)
            falls.append(curr_fall_list)
                        
            regex = "[0-9]+"
            player_given_damage = fun.readInput(f'Enter P{curr_player_index+1} given damage (without percent): ', regex)
            given_damages.append(int(player_given_damage))

            regex = "[0-9]+"
            player_taken_damage = fun.readInput(f'Enter P{curr_player_index+1} taken damage (without percent): ', regex)
            taken_damages.append(int(player_taken_damage))

            cv2.destroyAllWindows()
        else:
            #--- GET PLAYER FALLS ---#
            curr_fall_list = []
            fall_icon_px = res.LEFT_EDGE[players_amount-2][curr_player_index] + res.FALL_ICON_LD[players_amount-2]
            fall_icon_py = anchor_point + res.FALL_ICON_YO[players_amount-2]
            for j in range(custom.LIVES):
                fall_image = fun.submat(second_data, fall_icon_py, int(fall_icon_px), res.FALL_ICON_SIZE[players_amount-2], res.FALL_ICON_SIZE[players_amount-2])
                fall_image = cv2.resize(fall_image, (50,50))
                killer = fun.getClosestPlayerByIcon(fall_image, [res.CHARACTER_ICONS[index] for index in character_indices], res.BACKGROUND_ICONS_BGRS[types[curr_player_index]])
                if(killer == -1):
                    break
                if(killer == curr_player_index):
                    fun.showImage(cv2.resize(second_data, (640, 360)), f'match #{match_index}, second screenshot', 20)
                    print(f'Couldn\'t read P{curr_player_index+1}\'s falls properly. Please, add it manually.')

                    regex = "([" + "".join(map(str, range(1, curr_player_index+1))) + "".join(map(str, range(curr_player_index+2, players_amount+1))) + "](,[" + "".join(map(str, range(1, curr_player_index+1))) + "".join(map(str, range(curr_player_index+2, players_amount+1))) + "]){0," + str(custom.LIVES - 1) + "})?"
                    player_falls_string = fun.readInput(f'Enter P{curr_player_index+1} falls (separated by commas): ', regex)
                    curr_fall_list = []
                    for j in range(len(player_falls_string)):
                        if(j % 2 == 0):
                            fall_list.append(int(player_falls_string[j]))
                    
                    cv2.destroyAllWindows()
                    break

                curr_fall_list.append(killer + 1)
                fall_icon_px += res.FALL_ICON_SEP

            #--- GET PLAYER SELFDESTRUCTS ---#
            digit_y = anchor_point + res.SELFDESTR_YO
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD[language]
            digit_image = fun.submat(second_data, digit_y, digit_x, res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            curr_selfdestruct = fun.getMinMaxDigit(digit_image, res.MIN_SMALL_DIGIT_IMAGES, res.MAX_SMALL_DIGIT_IMAGES)
            if(curr_selfdestruct == -1):
                fun.showImage(cv2.resize(second_data, (640, 360)), f'match #{match_index}, second screenshot', 20)
                print(f'Couldn\'t read P{curr_player_index+1}\'s selfdestructions properly. Please, add it manually.')
                
                regex = f'[0-{custom.LIVES}]'
                curr_selfdestruct = int(fun.readInput(f'Enter P{curr_player_index+1} selfdestructions: ', regex))
                
                cv2.destroyAllWindows()

            for j in range(curr_selfdestruct):
                curr_fall_list.append(curr_player_index+1)

            #--- FALLS AND SELFDESTRUCTS - ERROR CHECKING ---#
            if(positions[curr_player_index] != 1 and len(curr_fall_list) != custom.LIVES):
                fun.showImage(cv2.resize(second_data, (640, 360)), f'match #{match_index}, second screenshot', 20)
                print(f'P{curr_player_index+1} (not in first place) died a number of times different from {custom.LIVES} ({curr_fall_list}). Please, add their data manually.')
                
                regex = f'[0-{custom.LIVES}]'
                curr_selfdestruct = int(fun.readInput(f'Enter P{curr_player_index+1} selfdestructions: ', regex))
                
                cv2.destroyAllWindows()
                error_message = f'P{curr_player_index+1} (not in first place) died a number of times different from {custom.LIVES} ({curr_fall_list})'
                raise fun.InvalidData
            if(positions[curr_player_index] == 1 and len(curr_fall_list) >= custom.LIVES):
                error_message = f'P{curr_player_index+1} (in first place) died more than {custom.LIVES-1} times ({curr_fall_list})'
                raise fun.InvalidData
            
            falls.append(curr_fall_list)


            #--- GET PLAYER GIVEN DAMAGE ---#
            given_damage = 0
            digit_y = anchor_point + res.GVN_DMG_YO
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.DMG_RD[language]
            digit = 0
            decimal_place = 1
            while(digit != -1):
                digit_image = fun.submat(second_data, digit_y, int(digit_x), res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getMinMaxDigit(digit_image, res.MIN_SMALL_DIGIT_IMAGES, res.MAX_SMALL_DIGIT_IMAGES)
                if(digit != -1):
                    given_damage += digit*decimal_place
                digit_x -= res.SMALL_DIGIT_SEP
                decimal_place *= 10
            given_damages.append(given_damage)
            
            #--- GET PLAYER TAKEN DAMAGE ---#
            taken_damage = 0
            digit_y = anchor_point + res.TKN_DMG_YO
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.DMG_RD[language]
            digit = 0
            decimal_place = 1
            while(digit != -1):
                digit_image = fun.submat(second_data, digit_y, int(digit_x), res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getMinMaxDigit(digit_image, res.MIN_SMALL_DIGIT_IMAGES, res.MAX_SMALL_DIGIT_IMAGES)
                if(digit != -1):
                    taken_damage += digit*decimal_place
                digit_x -= res.SMALL_DIGIT_SEP
                decimal_place *= 10
            taken_damages.append(taken_damage)
    
    #--- SECOND DATA - ERROR CHECKING ---#
    taken_given_dmg_difference = 0
    for curr_player_index in range(players_amount):
        taken_given_dmg_difference += taken_damages[curr_player_index] - given_damages[curr_player_index]
    if((taken_given_dmg_difference < 0 or taken_given_dmg_difference >= custom.TAKEN_GIVEN_DMG_THRESHOLD) and (("Ice Climbers" in [res.CHARACTER_INFOS[index][0] for index in character_indices]) == False)):
        error_message = f'too big of a difference between total taken damage and total given damage ({taken_given_dmg_difference}, taken damages: {taken_damages}, given damages: {given_damages})'
        raise fun.InvalidData


    #--- CONVERT MATCH DATA TO A STRING ---#
    output_strings.append(fun.convertMatchToString(dirs[2*match_index], [res.CHARACTER_INFOS[index][0] for index in character_indices], positions, times, falls, given_damages, taken_damages))


#--- WRITE OUTPUT ---#
output_file = open(custom.OUTPUT_PATH, 'w')
for match in output_strings[:-1]:
    output_file.write(match)
    output_file.write("\n")
output_file.write(output_strings[-1])
output_file.close()


#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
input("Press Enter to continue...")