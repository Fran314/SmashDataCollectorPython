from cv2 import cv2
import re
import numpy

class InvalidData(Exception):
    pass
class Skip(Exception):
    pass
class SkipAll(Exception):
    pass


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