import os
from cv2 import cv2
import numpy
import time

img1 = cv2.imread(r'E:\Archivio\Programmazione\VSCode\SmashDataCollector\readme_images\2021010119113300_c.jpg')
img2 = cv2.imread(r'E:\Archivio\Programmazione\VSCode\SmashDataCollector\readme_images\2021010119115300_c.jpg')

t = time.time()
for i in range(1000):
    numpy.sum(cv2.absdiff(img1, img2))

print(f'elapsed time: {(time.time() - t):.3f} s')