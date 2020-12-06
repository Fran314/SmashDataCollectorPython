import time
import numpy
from cv2 import cv2


#--- INITIALIZATION ---#
t = time.time()
#---   ---#


#--- CONSTANTS ---#
# "Anchor point" = the top left corner of the subimage used to identify
#   the position of the data. In this case we're using the rectangle
#   containing the text "Autodistruzioni" as an anchor point

PLAYERS = 3

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

#--- ---#

#--- TEST VARIABLES ---#
positions = [3, 2, 1]
characters = ["NESS", "CLOUD", "TOON LINK"]
#--- ---#


#stencil_source = r'D:\Utente\Desktop\stencil.png'
#data_source = r'D:\Utente\Desktop\data.jpg'
data_source = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\data_second.jpg'
ap_stencil_sources = []
for i in range(PLAYERS):
    ap_stencil_sources.append(r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\G' + str(i+1) + r'_ap_stencil.png')

data = cv2.imread(data_source, flags=cv2.IMREAD_UNCHANGED)
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)

ap_stencils = []
for i in range(PLAYERS):
    image = cv2.imread(ap_stencil_sources[i], flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    ap_stencils.append(image)

anchor_points = []
for i in range(PLAYERS):
    anchor_points.append(getAnchorPoint(data, AP_Xs[i], ap_stencils[i]))

for i in range(PLAYERS):
    print(f'G{i+1} anchor point at {anchor_points[i]}')


#--- CONCLUSION ---#
print("elapsed time: %.3f s" % (time.time() - t))
#--- ---#