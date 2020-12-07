import time
import numpy
import sys
import os
import pytesseract
from cv2 import cv2

#--- CUSTOMIZEABLE ---#
data_path = r'C:\Users\franc\Documents\VSCode\SmashDataAnalyzer\data'
res_path = r''
output_path = r''
#--- ---#

#--- FUNCTIONS ---#
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

def showImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    cv2.imshow("image", image)
    cv2.waitKey(0)
#--- ---#

dirs = mergeSort(os.listdir(data_path))
if(len(dirs) % 2 != 0):
    print("Odd number of images present in data. Ignoring the last image.")
    dirs = dirs[:-1]

for match_index in range(int(len(dirs)/2)):
    first_data = cv2.imread(os.path.join(data_path, dirs[2*match_index]), flags=cv2.IMREAD_UNCHANGED)
    first_data = cv2.cvtColor(first_data, cv2.COLOR_BGR2RGBA)

    second_data = cv2.imread(os.path.join(data_path, dirs[2*match_index + 1]), flags=cv2.IMREAD_UNCHANGED)
    second_data = cv2.cvtColor(second_data, cv2.COLOR_BGR2RGBA)

