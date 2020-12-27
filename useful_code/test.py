import time
import numpy
import pytesseract
from cv2 import cv2
import re
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# pylint: disable=import-error
import functions as fun

names = fun.readTSV(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\characters_info.tsv')
print(names)
print(1 - numpy.ones((2,3,4)))