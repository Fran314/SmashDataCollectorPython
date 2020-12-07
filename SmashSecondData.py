import time
import numpy
import pytesseract
from cv2 import cv2
import os


#--- INITIALIZATION ---#
t = time.time()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
#---   ---#


#--- CONSTANTS ---#
ICONS_FOLDER = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\res\icons'

IMAGE_POLARIZATION_TRESHOLD = 120

PLAYERS = 3

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

DAMAGE_OFF_X = 272
GIVEN_DMG_OFF_Y = 96
TAKEN_DMG_OFF_Y = 167
DAMAGE_WIDTH = 84
DAMAGE_HEIGHT = 37
DAMAGE_RIGHT_BORDER_WIDTH = 17
#---   ---#


#--- FUNCTIONS ---#
def polarizeImage(image_to_polarize):
    height = image_to_polarize.shape[0]
    width = image_to_polarize.shape[1]

    for i in range(height):
        for j in range(width):
            if(numpy.linalg.norm(image_to_polarize[i,j] - numpy.array([255, 255, 255, 255])) > IMAGE_POLARIZATION_TRESHOLD):
                image_to_polarize[i,j] = numpy.array([0, 0, 0, 255])

    return image_to_polarize

def imageDistance(arg0, arg1):
    shape0 = arg0.shape
    shape1 = arg1.shape
    if(shape0 != shape1):
        if(len(shape0) == 1):
            norm0 = numpy.linalg.norm(arg0)
        else:
            norm0 = numpy.linalg.norm(arg0[:,:])
        
        if(len(shape1) == 1):
            norm1 = numpy.linalg.norm(arg1)
        else:
            norm1 = numpy.linalg.norm(arg1[:,:])

        return numpy.max(2*norm0, 2*norm1)

    distance = 0
    distance += numpy.linalg.norm(numpy.multiply((arg0[:,:,0]/1.0 - arg1[:,:,0]/1.0), arg0[:,:,3]/255.0))
    distance += numpy.linalg.norm(numpy.multiply((arg0[:,:,1]/1.0 - arg1[:,:,1]/1.0), arg0[:,:,3]/255.0))
    distance += numpy.linalg.norm(numpy.multiply((arg0[:,:,2]/1.0 - arg1[:,:,2]/1.0), arg0[:,:,3]/255.0))

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
            path = os.path.join(ICONS_FOLDER, folderizeName(characters[i]), str(j+1) + ".png")
            character_image = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
            character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2RGBA)
            distance = imageDistance(character_image, image)
            character_distances.append(distance)
        distances.append(numpy.min(character_distances))
    return numpy.argmin(distances) - 1

def showImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    cv2.imshow("image", image)
    cv2.waitKey(0)
#--- ---#


#--- TEST VARIABLES ---#
positions = [3, 2, 1]
characters = ["NESS", "CLOUD", "TOON LINK"]
#--- ---#


#stencil_source = r'D:\Utente\Desktop\stencil.png'
#data_source = r'D:\Utente\Desktop\data.jpg'
data_source = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\res\data_second.jpg'
data = cv2.imread(data_source, flags=cv2.IMREAD_UNCHANGED)
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)

ap_stencils = []
for i in range(PLAYERS):
    ap_stencil_path = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\res\ap_stencils\G' + str(i+1) + r'_ap_stencil.png'
    image = cv2.imread(ap_stencil_path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    ap_stencils.append(image)

null_images = []
for i in range(PLAYERS):
    null_image_path = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\res\null_images\G' + str(i+1) + r'_null_image.png'
    image = cv2.imread(null_image_path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    null_images.append(image)

anchor_points = []
for i in range(PLAYERS):
    anchor_points.append(getAnchorPoint(data, AP_Xs[i], ap_stencils[i]))

kills = []
for i in range(PLAYERS):
    kill_string = []
    kill_icon_x = AP_Xs[i] + KILLS_AC_OFF_X
    kill_icon_y = anchor_points[i] + KILLS_AC_OFF_Y
    kill_image = data[kill_icon_y : kill_icon_y + KILL_ICON_SIZE, kill_icon_x : kill_icon_x + KILL_ICON_SIZE]
    killer = getClosestPlayer(kill_image, null_images[i], characters)
    kill_string.append(killer + 1)
    
    kill_icon_x += KILLS_OFFSET
    kill_image = data[kill_icon_y : kill_icon_y + KILL_ICON_SIZE, kill_icon_x : kill_icon_x + KILL_ICON_SIZE]
    killer = getClosestPlayer(kill_image, null_images[i], characters)
    kill_string.append(killer + 1)

    kill_icon_x += KILLS_OFFSET
    kill_image = data[kill_icon_y : kill_icon_y + KILL_ICON_SIZE, kill_icon_x : kill_icon_x + KILL_ICON_SIZE]
    killer = getClosestPlayer(kill_image, null_images[i], characters)
    kill_string.append(killer + 1)

    kills.append(kill_string)

given_damages = []
taken_damages = []
for i in range(PLAYERS):
    damage_x = AP_Xs[i] + DAMAGE_OFF_X
    given_dmg_y = anchor_points[i] + GIVEN_DMG_OFF_Y
    given_damage_rect = data[given_dmg_y : given_dmg_y + DAMAGE_HEIGHT, damage_x : damage_x + DAMAGE_WIDTH]
    for j in range(DAMAGE_RIGHT_BORDER_WIDTH):
        for h in range(DAMAGE_HEIGHT):
            given_damage_rect[h, -j] = given_damage_rect[4, 1]
    given_damage_rect = polarizeImage(given_damage_rect)
    given_damages.append(normalizeDamage(pytesseract.image_to_string(given_damage_rect)))

    taken_dmg_y = anchor_points[i] + TAKEN_DMG_OFF_Y
    taken_damage_rect = data[taken_dmg_y : taken_dmg_y + DAMAGE_HEIGHT, damage_x : damage_x + DAMAGE_WIDTH]
    for j in range(DAMAGE_RIGHT_BORDER_WIDTH):
        for h in range(DAMAGE_HEIGHT):
            taken_damage_rect[h, -j] = taken_damage_rect[4, 1]
    taken_damage_rect = polarizeImage(taken_damage_rect)
    taken_damages.append(normalizeDamage(pytesseract.image_to_string(taken_damage_rect)))


for i in range(PLAYERS):
    print(f'G{i+1} killed by {kills[i]}')
    print(f'given damage: {given_damages[i]}')
    print(f'taken damage: {taken_damages[i]}')
    print()

#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
#--- ---#