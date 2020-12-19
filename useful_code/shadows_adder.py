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

def deshadow(image):
    temp_alphas = numpy.copy(image[:,:,3])
    image[:,:,3] = image[:,:,3]*2.0 - 255.0*numpy.ones_like(image[:,:,3])
    image[:,:,3][temp_alphas <= 127] = 0
    return image

def shadowOf(image):
    x_offset = 24
    y_offset = 18
    shifted = numpy.zeros((image.shape[0] + y_offset, image.shape[1] + x_offset, 4))
    shifted[y_offset:, x_offset:, :] = image
    to_return = numpy.zeros(image.shape)
    to_return[:,:,3] = (108.0*shifted[:-y_offset,:-x_offset,3])/255.0
    return to_return

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
    return to_return.astype(numpy.float32)


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

# characters = readList(r'D:\Utente\Desktop\lista_personaggi.csv')
characters = ["sephiroth"]

clean_artworks_folder = r'D:\Utente\Desktop\smash\clean_artworks'
shadowed_artworks_folder = r'D:\Utente\Desktop\smash\shadowed_artworks'

total_time = time.time()
for character in characters:
    character_time = time.time()
    print(character, end='')
    for i in range(8):
        character_image = readImage(os.path.join(clean_artworks_folder, character, str(i) + ".png"))
        character_image = deshadow(character_image)
        try:
            os.mkdir(os.path.join(shadowed_artworks_folder, character))
        except Exception:
            pass
        saveImage(overlayImages(character_image, shadowOf(character_image)), os.path.join(shadowed_artworks_folder, character, str(i) + ".png"))
    print(f', elapsed time: {(time.time() - character_time):.3f} s')
print(f'\ntotal elapsed time: {(time.time() - total_time):.3f} s')