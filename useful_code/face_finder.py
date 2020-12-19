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
    character_time = time.time()
    print(character, end='')
    images = os.listdir(os.path.join(faces_folder, character))
    for face_image_file in images:
        face_image = readImage(os.path.join(faces_folder, character, face_image_file))
        artwork_image = readImage(os.path.join(clean_artworks_folder, character, face_image_file[0] + ".png"))
        player_sample_pixel = face_image[-3:-10]
        player = numpy.argmin([norm(player_sample_pixel - reference_pixel) for reference_pixel in SAMPLE_G])
        background_image = numpy.full_like(artwork_image, 255)
        background_image[:,:,0] = numpy.full((artwork_image.shape[0], artwork_image.shape[1]), RED_G[player])
        background_image[:,:,1] = numpy.full((artwork_image.shape[0], artwork_image.shape[1]), GREEN_G[player])
        background_image[:,:,2] = numpy.full((artwork_image.shape[0], artwork_image.shape[1]), BLUE_G[player])
        artwork_image = overlayImages(artwork_image, background_image)

        scale_range = numpy.linspace(0.707, 1.414, 50)
        width_range = (artwork_image.shape[1] * numpy.multiply(scale_range,scale_range)).astype(int)
        min_value = float("inf")
        approx_index = 0
        for i in range(len(width_range)):
            width = width_range[i]
            height = int((width * artwork_image.shape[0]) / artwork_image.shape[1])
            scaled_artwork = cv2.resize(artwork_image, (width, height))
            template_map = cv2.matchTemplate(scaled_artwork, face_image, cv2.TM_SQDIFF, None, mask_image[player])

            curr_min = numpy.min(template_map)
            if(curr_min < min_value):
                approx_index = i
                min_value = curr_min

        approx_index = max(1, min(approx_index, len(width_range)-2))
        min_value = float("inf")
        final_width = 0
        final_height = 0
        final_x = 0
        final_y = 0
        for width in range(width_range[approx_index - 1], width_range[approx_index + 1]):
            height = int((width * artwork_image.shape[0]) / artwork_image.shape[1])
            scaled_artwork = cv2.resize(artwork_image, (width, height))
            template_map = cv2.matchTemplate(scaled_artwork, face_image, cv2.TM_SQDIFF, None, mask_image[player])

            curr_min_index = numpy.argmin(template_map)
            curr_min_x = curr_min_index % template_map.shape[1]
            curr_min_y = int((curr_min_index - curr_min_x) / template_map.shape[1])
            if(template_map[curr_min_y,curr_min_x] < min_value):
                final_width = width
                final_height = height
                final_x = curr_min_x
                final_y = curr_min_y
                min_value = template_map[curr_min_y,curr_min_x]
        
        for c in face_image_file[:-4]:
            artwork_image = readImage(os.path.join(clean_artworks_folder, character, c + ".png"))
            try:
                os.mkdir(os.path.join(output_artworks_folder, character))
            except Exception:
                pass
            saveImage(cv2.resize(artwork_image, (final_width, final_height))[final_y : final_y + face_image.shape[0], final_x : final_x + face_image.shape[1]], os.path.join(output_artworks_folder, character, c + ".png"))
        
    print(f', elapsed time: {formatTime(time.time() - character_time)}, ETL: {formatTime(((time.time() - total_time) * (len(characters) - character_index-1)) / (character_index+1))}')
print(f'\ntotal elapsed time: {formatTime(time.time() - total_time)}')