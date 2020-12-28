import time
import numpy
import pytesseract
from cv2 import cv2
import re
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# pylint: disable=import-error
import functions as fun

first = cv2.imread(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\faces\bayonetta\0246.png')
second = cv2.imread(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\faces\bayonetta\1357.png')
mask = (cv2.imread(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\23-lose_mask.png') / 255).astype(numpy.uint8)
mask2 = (cv2.cvtColor(cv2.imread(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\23-lose_mask.png'), cv2.COLOR_BGR2GRAY) / 255).astype(numpy.uint8)

t = time.time()
for i in range(1000):
    sum = numpy.sum(numpy.multiply(cv2.absdiff(first, second), mask))
print(f'elapsed time: {(time.time() - t):.3f} s')
print(sum)

t = time.time()
for i in range(1000):
    sum = numpy.sum(numpy.where(mask > 0.5, cv2.absdiff(first, second), [0, 0, 0]))
print(f'elapsed time: {(time.time() - t):.3f} s')
print(sum)