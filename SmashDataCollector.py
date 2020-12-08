import time
import numpy
import sys
import os
import pytesseract
from cv2 import cv2

#--- CUSTOMIZEABLE ---#
data_path = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\data'
res_path = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\res'
output_path = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer'
#--- ---#


#--- INITIALIZATION ---#
#os.system('mode con: cols=88 lines=30')
t = time.time()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
#---   ---#


#--- CLASSES ---#
class InvalidData(Exception):
    pass
#--- ---#


#--- CONSTANTS ---#
PLAYERS = 3 # Number of players

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

POLARIZATION_THRESHOLD = 40

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


# "Anchor point" = the top left corner of the subimage used to identify
#   the position of the data. In this case we're using the rectangle
#   containing the text "Autodistruzioni" as an anchor point
AP_Xs = [43, # X position of the anchor point for G1
        462, # X position of the anchor point for G2
        880] # X position of the anchor point for G3

KILL_ICON_SIZE = 29
KILLS_AC_OFF_X = 14 # Difference in X coordinates from the anchor point to the first kill icon
KILLS_AC_OFF_Y = -44 # Difference in Y coordinates from the anchor point to the first kill icon
KILLS_OFFSET = 33 # Offset on X coordinates between each kill icon

DAMAGE_OFF_Xs = [281, 300, 320]
GIVEN_DMG_OFF_Y_DIG = 103
TAKEN_DMG_OFF_Y_DIG = 174
DIGIT_WIDTH = 20
DIGIT_HEIGHT = 25
#--- ---#


#--- FUNCTIONS ---#
def polarizeImage(image_to_polarize, treshold):
    height = image_to_polarize.shape[0]
    width = image_to_polarize.shape[1]

    to_return = numpy.copy(image_to_polarize)

    for i in range(height):
        for j in range(width):
            if(numpy.linalg.norm(image_to_polarize[i,j] - numpy.array([255, 255, 255, 255])) > treshold):
                to_return[i,j] = numpy.array([0, 0, 0, 255])

    return to_return


def getName(data, pos_x, is_winner):
    if(is_winner == 1):
        name_rect = data[W_NAME_Y : (W_NAME_Y + NAME_HEIGHT), pos_x : (pos_x + NAME_WIDTH)]
    else:
        name_rect = data[L_NAME_Y : (L_NAME_Y + NAME_HEIGHT), pos_x : (pos_x + NAME_WIDTH)]

    name_rect = polarizeImage(name_rect, POLARIZATION_THRESHOLD)
    name = normalizeName(pytesseract.image_to_string(name_rect))
    character_name_errors = numpy.zeros(len(CHARACTER_NAMES), dtype=int)
    for i in range(len(CHARACTER_NAMES)):
        character_name_errors[i] = optimalAlignError(name, CHARACTER_NAMES[i][0])
    
    return CHARACTER_NAMES[numpy.argmin(character_name_errors)][1]


def getTimeRectangle(data, pos_x):
    return cv2.resize(polarizeImage(data[TIME_Y : (TIME_Y + TIME_HEIGHT), pos_x : (pos_x + TIME_WIDTH)], POLARIZATION_THRESHOLD), (2*TIME_WIDTH, 2*TIME_HEIGHT))


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


def imageDistance(stencil, image):
    shape0 = stencil.shape
    shape1 = image.shape
    if(shape0 != shape1):
        if(len(shape0) == 1):
            norm0 = numpy.linalg.norm(stencil)
        else:
            norm0 = numpy.linalg.norm(stencil[:,:])
        
        if(len(shape1) == 1):
            norm1 = numpy.linalg.norm(image)
        else:
            norm1 = numpy.linalg.norm(image[:,:])

        return numpy.max(2*norm0, 2*norm1)

    distance = 0
    distance += numpy.linalg.norm(numpy.multiply((stencil[:,:,0]/1.0 - image[:,:,0]/1.0), stencil[:,:,3]/255.0))
    distance += numpy.linalg.norm(numpy.multiply((stencil[:,:,1]/1.0 - image[:,:,1]/1.0), stencil[:,:,3]/255.0))
    distance += numpy.linalg.norm(numpy.multiply((stencil[:,:,2]/1.0 - image[:,:,2]/1.0), stencil[:,:,3]/255.0))

    return distance


def getAnchorPoint(data, pos_x, stencil):
    sh = stencil.shape[0] # stencil height
    sw = stencil.shape[1] # stencil width
    dh = data.shape[0] # data image height

    min_norm = imageDistance(stencil, data[0:sh, pos_x:pos_x+sw])
    pos_y = 0

    for i in range(dh - sh):
        curr_norm = imageDistance(stencil, data[i:i+sh, pos_x:pos_x+sw])
        if(curr_norm < min_norm):
            min_norm = curr_norm
            pos_y = i
            
    return pos_y


def folderizeName(arg):
    to_return = ""
    for c in arg:
        if(ord(c) >= 97 and ord(c) <= 122):
            to_return += c
        elif(ord(c) >= 65 and ord(c) <= 90):
            to_return += chr(ord(c) + 32)
    
    return to_return


def normalizeDamage(arg):
    to_return = ""
    for c in arg:
        if((ord(c) >= 48 and ord(c) <= 57)):
            to_return += c
    
    return to_return


def getClosestPlayer(image, null_image, characters):
    distances = [imageDistance(null_image, image)]
    for i in range(len(characters)):
        character_distances = []
        for j in range(8):
            path = os.path.join(res_path, "icons", folderizeName(characters[i]), str(j+1) + ".png")
            character_image = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
            character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2RGBA)
            distance = imageDistance(character_image, image)
            character_distances.append(distance)
        distances.append(numpy.min(character_distances))
    return numpy.argmin(distances) - 1


def getClosestDigit(image):
    distances = []
    image = polarizeImage(image, POLARIZATION_THRESHOLD)
    for i in range(10):
        path = os.path.join(res_path, "small_digits",  str(i) + ".png")
        digit_image = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
        digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2RGBA)
        distances.append(imageDistance(digit_image, image))
    
    digit = numpy.argmin(distances)
    if(digit == 6 or digit == 8):
        if(image[8,16,0] > 127):
            return 8
        else:
            return 6
    return digit


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
        
    times_sec = convertTimesToSec(times)

    for i in range(2, len(times)):
        if(positions[order[i]] == i+1 and times_sec[order[i]] >= times_sec[order[i-1]]):
            return False
        if(positions[order[i]] != i+1 and times_sec[order[i]] != times_sec[order[i-1]]):
            return False
    
    return True

def convertTimesToSec(times):
    times_sec = []
    first_pos = 0
    for i in range(len(times)):
        if(times[i] == ""):
            times_sec.append(0)
            first_pos = i
        else:
            times_sec.append(60 * int(times[i][0]) + int(times[i][2:3]))

    times_sec[first_pos] = numpy.max(times_sec)
    return times_sec
            

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


def showImage(image, text = "image", delay = 0):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    cv2.namedWindow(text)
    cv2.moveWindow(text, 20, 20)
    cv2.imshow(text, image)
    cv2.waitKey(delay)
#--- ---#


dirs = mergeSort(os.listdir(data_path))
if(len(dirs) % 2 != 0):
    print("Odd number of images present in data. Ignoring the last image.")
    dirs = dirs[:-1]

ap_stencils = []
for i in range(PLAYERS):
    ap_stencil_path = res_path + r'\ap_stencils\G' + str(i+1) + r'_ap_stencil.png'
    image = cv2.imread(ap_stencil_path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    ap_stencils.append(image)

null_images = []
for i in range(PLAYERS):
    null_image_path = res_path + r'\null_images\G' + str(i+1) + r'_null_image.png'
    image = cv2.imread(null_image_path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    null_images.append(image)

output_file = open(os.path.join(output_path, "output.txt"), 'w')
for match_index in range(int(len(dirs)/2)):
    print(f'match {match_index+1}/{int(len(dirs)/2)}')
    first_data = cv2.imread(os.path.join(data_path, dirs[2*match_index]), flags=cv2.IMREAD_UNCHANGED)
    first_data = cv2.cvtColor(first_data, cv2.COLOR_BGR2RGBA)

    second_data = cv2.imread(os.path.join(data_path, dirs[2*match_index + 1]), flags=cv2.IMREAD_UNCHANGED)
    second_data = cv2.cvtColor(second_data, cv2.COLOR_BGR2RGBA)

    #--- FIRST IMAGE ---#
    positions = []
    characters = []
    times = []
    try:
        for i in range(PLAYERS):
            col = first_data[SAMPLE_POINT_Y, SAMPLE_POINT_Xs[i]]
            positions.append(numpy.argmin(numpy.array([numpy.linalg.norm(col - FIRST_COL), numpy.linalg.norm(col - SECOND_COL), numpy.linalg.norm(col - THIRD_COL)]))+1)

        for i in range(PLAYERS):
            characters.append(getName(first_data, NAME_Xs[i], positions[i]))

        for i in range(PLAYERS):
            time_rect = getTimeRectangle(first_data, TIME_Xs[i])
            times.append(normalizeTime(pytesseract.image_to_string(time_rect)))

        for i in range(PLAYERS):
            if(positions[i] < 1 or positions[i] > PLAYERS):
                raise InvalidData
            if(isValidTime(times[i]) == False):
                raise InvalidData
            if(isValidFirstData(positions, times) == False):
                raise InvalidData
    except InvalidData:
        print("Unable to read data. Please insert the data manually.")
        cv2.destroyAllWindows()
        showImage(cv2.resize(first_data, (640, 360)), f'match {match_index+1}/{int(len(dirs)/2)}, first screenshot', 20)
        valid_data = False
        while(valid_data == False):
            positions = []
            characters = []
            times = []
            for i in range(PLAYERS):
                valid_input = False
                player_character = ""
                player_data = ""
                while(valid_input == False):
                    player_character = input(f'insert G{i+1} character: ')
                    for j in range(len(CHARACTER_NAMES)):
                        if(player_character.upper() == CHARACTER_NAMES[j][0]):
                            valid_input = True
                            player_character = CHARACTER_NAMES[j][1]
                    if(valid_input == False):
                        print("Invalid input")
                        continue
                
                characters.append(player_character)

                valid_input = False
                while(valid_input == False):
                    valid_input = True
                    player_data = input(f'insert G{i+1} position and time ([pos]  [m:ss]): ')
                    if(len(player_data) != 6 and len(player_data) != 1):
                        valid_input = False
                        print("Invalid input")
                        continue
                    if(ord(player_data[0]) < 49 or ord(player_data[0]) > 48 + PLAYERS):
                        valid_input = False
                        print("Invalid input")
                        continue
                    if(player_data[0] == '1'):
                        if(len(player_data) != 1):
                            valid_input = False
                            print("Invalid input")
                            continue
                    else:
                        if(len(player_data) != 6):
                            valid_input = False
                            print("Invalid input")
                            continue
                        if(isValidTime(player_data[2:]) == False):
                            valid_input = False
                            print("Invalid input")
                            continue
                positions.append(int(player_data[0]))
                if(player_data[0] != '1'):
                    times.append(player_data[2:])
                else:
                    times.append("")
            
            valid_data = isValidFirstData(positions, times)
            if(valid_data == False):
                print("The data inserted has an error or is self-contradictory in some way. Please check and insert the data again.")
        print("The data inserted is valid!")
        cv2.destroyAllWindows()
    #--- ---#

    #--- SECOND IMAGE ---#
    anchor_points = []
    for i in range(PLAYERS):
        anchor_points.append(getAnchorPoint(second_data, AP_Xs[i], ap_stencils[i]))

    kills = []
    for i in range(PLAYERS):
        kill_string = []
        kill_icon_x = AP_Xs[i] + KILLS_AC_OFF_X
        kill_icon_y = anchor_points[i] + KILLS_AC_OFF_Y
        kill_image = second_data[kill_icon_y : kill_icon_y + KILL_ICON_SIZE, kill_icon_x : kill_icon_x + KILL_ICON_SIZE]
        killer = getClosestPlayer(kill_image, null_images[i], characters)
        if(killer != -1):
            kill_string.append(killer + 1)
        
        kill_icon_x += KILLS_OFFSET
        kill_image = second_data[kill_icon_y : kill_icon_y + KILL_ICON_SIZE, kill_icon_x : kill_icon_x + KILL_ICON_SIZE]
        killer = getClosestPlayer(kill_image, null_images[i], characters)
        if(killer != -1):
            kill_string.append(killer + 1)

        kill_icon_x += KILLS_OFFSET
        kill_image = second_data[kill_icon_y : kill_icon_y + KILL_ICON_SIZE, kill_icon_x : kill_icon_x + KILL_ICON_SIZE]
        killer = getClosestPlayer(kill_image, null_images[i], characters)
        if(killer != -1):
            kill_string.append(killer + 1)

        kills.append(kill_string)

    given_damages = []
    taken_damages = []
    for i in range(PLAYERS):
        given_damage = 0
        digit_y = anchor_points[i] + GIVEN_DMG_OFF_Y_DIG
        digit_x = AP_Xs[i] + DAMAGE_OFF_Xs[0]
        given_damage += getClosestDigit(second_data[digit_y : digit_y + DIGIT_HEIGHT, digit_x : digit_x + DIGIT_WIDTH])*100
        digit_x = AP_Xs[i] + DAMAGE_OFF_Xs[1]
        given_damage += getClosestDigit(second_data[digit_y : digit_y + DIGIT_HEIGHT, digit_x : digit_x + DIGIT_WIDTH])*10
        digit_x = AP_Xs[i] + DAMAGE_OFF_Xs[2]
        given_damage += getClosestDigit(second_data[digit_y : digit_y + DIGIT_HEIGHT, digit_x : digit_x + DIGIT_WIDTH])
        given_damages.append(given_damage)
        
        taken_damage = 0
        digit_y = anchor_points[i] + TAKEN_DMG_OFF_Y_DIG
        digit_x = AP_Xs[i] + DAMAGE_OFF_Xs[0]
        taken_damage += getClosestDigit(second_data[digit_y : digit_y + DIGIT_HEIGHT, digit_x : digit_x + DIGIT_WIDTH])*100
        digit_x = AP_Xs[i] + DAMAGE_OFF_Xs[1]
        taken_damage += getClosestDigit(second_data[digit_y : digit_y + DIGIT_HEIGHT, digit_x : digit_x + DIGIT_WIDTH])*10
        digit_x = AP_Xs[i] + DAMAGE_OFF_Xs[2]
        taken_damage += getClosestDigit(second_data[digit_y : digit_y + DIGIT_HEIGHT, digit_x : digit_x + DIGIT_WIDTH])
        taken_damages.append(taken_damage)
    #--- ---#

    dirs[2*match_index]
    output_file.write("\"" + dirs[2*match_index][0:4] + "/" + dirs[2*match_index][4:6] + "/" + dirs[2*match_index][6:8] + "\", ")
    output_file.write("\"" + str(PLAYERS) + "\", ")
    for i in range(PLAYERS):
        output_file.write("\"" + characters[i] + "\", ")
        output_file.write("\"" + str(positions[i]) + "\", ")
        if(len(kills[i]) > 0):
            output_file.write("\"")
            for j in range(len(kills[i])):
                output_file.write(str(kills[i][j]))
                if(j < len(kills[i]) - 1):
                    output_file.write(",")
        output_file.write("\", ")
        output_file.write("\"" + str(given_damages[i]) + "\", ")
        output_file.write("\"" + str(taken_damages[i]) + "\", ")
        output_file.write("\"" + str(times[i]) + "\"")
        if(i < PLAYERS - 1):
            output_file.write(", ")
    if(match_index < len(dirs)/2 - 1):
        output_file.write("\n")

output_file.close


#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
#--- ---#