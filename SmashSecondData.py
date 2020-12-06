import time
import numpy
from cv2 import cv2
import os


#--- INITIALIZATION ---#
t = time.time()
#---   ---#


#--- CONSTANTS ---#
ICONS_FOLDER = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\icons'

PLAYERS = 3

# "Anchor point" = the top left corner of the subimage used to identify
#   the position of the data. In this case we're using the rectangle
#   containing the text "Autodistruzioni" as an anchor point
AP_Xs = [43, # X position of the anchor point for G1
        462, # X position of the anchor point for G2
        880] # X position of the anchor point for G3
#---   ---#


#--- DEFINITIONS ---#
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


def getClosestPlayer(image, null_image, characters):
    distances = [imageDistance(null_image, image)]
    for i in range(len(characters)):
        character_distances = []
        for j in range(8):
            path = os.path.join(ICONS_FOLDER, folderizeName(characters[i]), str(j+1) + ".png")
            character_image = cv2.imread(path)
            character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2RGBA)
            character_distances.append(imageDistance(character_image, image))
        distances.append(numpy.min(character_distances))
    return numpy.argmin(distances) - 1

#--- ---#


#--- TEST VARIABLES ---#
positions = [3, 2, 1]
characters = ["NESS", "CLOUD", "TOON LINK"]
#--- ---#


#stencil_source = r'D:\Utente\Desktop\stencil.png'
#data_source = r'D:\Utente\Desktop\data.jpg'
data_source = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\data_second.jpg'
data = cv2.imread(data_source, flags=cv2.IMREAD_UNCHANGED)
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)

ap_stencils = []
for i in range(PLAYERS):
    ap_stencil_path = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\G' + str(i+1) + r'_ap_stencil.png'
    image = cv2.imread(ap_stencil_path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    ap_stencils.append(image)

null_images = []
for i in range(PLAYERS):
    null_image_path = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\G' + str(i+1) + r'_null_image.png'
    image = cv2.imread(null_image_path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    null_images.append(image)

anchor_points = []
for i in range(PLAYERS):
    anchor_points.append(getAnchorPoint(data, AP_Xs[i], ap_stencils[i]))

kills = []
for i in range(PLAYERS):
    kill_string = []
    kill_image = data[anchor_points[i] - 44 : anchor_points[i] - 44 + 29, AP_Xs[i] + 14 : AP_Xs[i] + 14 + 29]
    killer = getClosestPlayer(kill_image, null_images[i], characters)
    if(killer != -1):
        kill_string.append(killer + 1)

    kill_image = data[anchor_points[i] - 44 : anchor_points[i] - 44 + 29, AP_Xs[i] + 14 + 33: AP_Xs[i] + 14 + 29 + 33]
    killer = getClosestPlayer(kill_image, null_images[i], characters)
    if(killer != -1):
        kill_string.append(killer + 1)

    kill_image = data[anchor_points[i] - 44 : anchor_points[i] - 44 + 29, AP_Xs[i] + 14 + 66: AP_Xs[i] + 14 + 29 + 66]
    killer = getClosestPlayer(kill_image, null_images[i], characters)
    if(killer != -1):
        kill_string.append(killer + 1)

    kills.append(kill_string)

for i in range(PLAYERS):
    print(f'G{i+1} anchor point at {anchor_points[i]}')
    print(f'killed by {kills[i]}')
    print()

#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
#--- ---#