import numpy
import sys
from cv2 import cv2
import math

def imageDistance(arg0, arg1):
    shape0 = arg0.shape
    shape1 = arg1.shape
    if(shape0 != shape1):
        return sys.float_info.max

    distance = 0
    distance += numpy.linalg.norm(numpy.multiply((arg0[:,:,0]/1.0 - arg1[:,:,0]/1.0), arg0[:,:,3]/255.0))
    distance += numpy.linalg.norm(numpy.multiply((arg0[:,:,1]/1.0 - arg1[:,:,1]/1.0), arg0[:,:,3]/255.0))
    distance += numpy.linalg.norm(numpy.multiply((arg0[:,:,2]/1.0 - arg1[:,:,2]/1.0), arg0[:,:,3]/255.0))

    return distance

#stencil_source = r'D:\Utente\Desktop\stencil.png'
#data_source = r'D:\Utente\Desktop\data.jpg'
stencil_source = r'C:\Users\franc\Desktop\stencil.png'
data_source = r'C:\Users\franc\Desktop\data.jpg'

stencil = cv2.imread(stencil_source, flags=cv2.IMREAD_UNCHANGED)
stencil = cv2.cvtColor(stencil, cv2.COLOR_BGR2RGBA)

data = cv2.imread(data_source, flags=cv2.IMREAD_UNCHANGED)
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)

sh = stencil.shape[0]
sw = stencil.shape[1]

dh = data.shape[0]
dw = data.shape[1]

treshold = 103*(math.sqrt(sh) + math.sqrt(sw))/2.0

#43, 377

min_norm_1 = imageDistance(stencil, data[1:1+sh, 43:43+sw])
min_norm_2 = imageDistance(stencil, data[1:1+sh, 43:43+sw])
min_pos_1 = 0
min_pos_2 = 0

for i in range(dh - sh):
    curr_norm = imageDistance(stencil, data[i:i+sh, 43:43+sw])
    if(curr_norm < min_norm_1):
        min_norm_1 = curr_norm
        min_pos_1 = i

for i in range(dh - sh):
    if(i != min_pos_1):
        curr_norm = imageDistance(stencil, data[i:i+sh, 43:43+sw])
        if(curr_norm < min_norm_2):
            min_norm_2 = curr_norm
            min_pos_2 = i

print("Min at %s, %s" %(min_pos_1, min_norm_1))
print("Cls at %s, %s" %(min_pos_2, min_norm_2))