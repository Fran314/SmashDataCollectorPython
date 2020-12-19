import time
import numpy
from numpy.linalg import norm
import sys
import os
import pytesseract
from cv2 import cv2
import re

def showImage(image, text = "image", delay = 0):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    cv2.namedWindow(text)
    cv2.moveWindow(text, 20, 20)
    cv2.imshow(text, image)
    cv2.waitKey(delay)
def readImage(path):
    image = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return image
def saveImage(image, path):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, image)


def readList(path):
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
    
    return lines


def overlayImages(foreground, background):
    width = numpy.min([foreground.shape[1], background.shape[1]])
    height = numpy.min([foreground.shape[0], background.shape[0]])
    to_return = numpy.zeros((height, width, 4))
    first_part = numpy.zeros((height, width, 3))
    second_part = numpy.zeros((height, width, 3))
    first_part[:,:,0] = numpy.multiply(foreground[0 : height, 0 : width, 0], foreground[0 : height, 0 : width, 3] / 255.0)
    first_part[:,:,1] = numpy.multiply(foreground[0 : height, 0 : width, 1], foreground[0 : height, 0 : width, 3] / 255.0)
    first_part[:,:,2] = numpy.multiply(foreground[0 : height, 0 : width, 2], foreground[0 : height, 0 : width, 3] / 255.0)
    second_part[:,:,0] = numpy.multiply(background[0 : height, 0 : width, 0], numpy.multiply(background[0 : height, 0 : width, 3] / 255.0, numpy.ones((height, width)) - (foreground[0 : height, 0 : width, 3]/ 255.0)))
    second_part[:,:,1] = numpy.multiply(background[0 : height, 0 : width, 1], numpy.multiply(background[0 : height, 0 : width, 3] / 255.0, numpy.ones((height, width)) - (foreground[0 : height, 0 : width, 3]/ 255.0)))
    second_part[:,:,2] = numpy.multiply(background[0 : height, 0 : width, 2], numpy.multiply(background[0 : height, 0 : width, 3] / 255.0, numpy.ones((height, width)) - (foreground[0 : height, 0 : width, 3]/ 255.0)))
    colors = first_part + second_part
    alphas = foreground[0 : height, 0 : width, 3] / 255.0 + numpy.multiply(background[0 : height, 0 : width, 3] / 255.0, numpy.ones((height, width)) - (foreground[0 : height, 0 : width, 3]/ 255.0))
    colors[:,:,0] = numpy.divide(colors[:,:,0], alphas, out=numpy.zeros_like(colors[:,:,0]), where=alphas!=0)
    colors[:,:,1] = numpy.divide(colors[:,:,1], alphas, out=numpy.zeros_like(colors[:,:,1]), where=alphas!=0)
    colors[:,:,2] = numpy.divide(colors[:,:,2], alphas, out=numpy.zeros_like(colors[:,:,2]), where=alphas!=0)
    to_return[:,:, 0:3] = colors
    to_return[:,:,3] = alphas*255
    return to_return.astype(numpy.uint8)


def formatTime(arg):
    arg = int(arg)
    seconds = arg % 60
    arg = int((arg - seconds)/60)
    minutes = arg % 60
    arg = int((arg - minutes)/60)
    hours = arg
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


RED_G = [188, 7, 232]
GREEN_G = [4, 102, 172]
BLUE_G = [0, 213, 9]

SAMPLE_G = [numpy.array([211, 51, 51, 255]), numpy.array([39, 108, 237, 255]), numpy.array([228, 170, 0, 255])]

characters = readList(r'D:\Utente\Desktop\lista_personaggi.csv')

faces_folder = r'D:\Utente\Desktop\smash\faces'
clean_artworks_folder = r'D:\Utente\Desktop\smash\shadowed_artworks'
output_artworks_folder = r'D:\Utente\Desktop\smash\coded_faces'

mask_image = [readImage(r'D:\Utente\Desktop\foreground_mask.png'), readImage(r'D:\Utente\Desktop\foreground_mask.png'), readImage(r'D:\Utente\Desktop\foreground_mask_G3.png')]

total_time = time.time()
for character_index in range(len(characters)):
    character = characters[character_index]
    if(os.path.isdir(os.path.join(faces_folder, character)) == False):
        continue
    values = [0, 0, 0, 0, 0, 0, 0, 0]
    images = os.listdir(os.path.join(faces_folder, character))
    for image in images:
        for c in image[:-4]:
            values[int(c)] += 1
    
    for i in range(8):
        if(values[i] != 1):
            print(f'Problems with {character}')
print(f'\ntotal elapsed time: {formatTime(time.time() - total_time)}')