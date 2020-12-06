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
G1_AP_X = 43 # Y position of the anchor point for G1
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


#stencil_source = r'D:\Utente\Desktop\stencil.png'
#data_source = r'D:\Utente\Desktop\data.jpg'
data_source = r'C:\Users\franc\Desktop\data.jpg'
G1_ap_stencil_source = r'C:\Users\franc\Desktop\G1_ap_stencil.png'

data = cv2.imread(data_source, flags=cv2.IMREAD_UNCHANGED)
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)

# G1 anchor point stencil
G1_ap_stencil = cv2.imread(G1_ap_stencil_source, flags=cv2.IMREAD_UNCHANGED)
G1_ap_stencil = cv2.cvtColor(G1_ap_stencil, cv2.COLOR_BGR2RGBA)

G1_ap = getAnchorPoint(data, G1_AP_X, G1_ap_stencil)

print("G1 anchor point at %s" %(G1_ap))


#--- CONCLUSION ---#
print("elapsed time: %.3f s" % (time.time() - t))
#--- ---#