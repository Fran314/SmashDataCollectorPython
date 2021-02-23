import time
import numpy
from numpy.linalg import norm
import sys
import os
import pytesseract
from cv2 import cv2
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# pylint: disable=import-error
import functions as fun
import resources as res


def formatTime(arg):
    arg = int(arg)
    seconds = arg % 60
    arg = int((arg - seconds)/60)
    minutes = arg % 60
    arg = int((arg - minutes)/60)
    hours = arg
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def deshadow(image):
    temp_alphas = numpy.copy(image[:,:,3])
    image[:,:,3] = image[:,:,3]*2.0 - 255.0*numpy.ones_like(image[:,:,3])
    image[:,:,3][temp_alphas <= 127] = 0
    return image

def shadowOf(image, x_offset, y_offset):
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


MANUAL = False

WIDTH_SAMPLES = 100

SAMPLE_G = [numpy.array([51, 51, 211]), numpy.array([237, 108, 39]), numpy.array([0, 170, 228]), numpy.array([41, 153, 12])]
characters = fun.readTSV(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\characters_info.tsv')

faces_folder = r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\23-faces'
artworks_folder = r'C:\Users\franc\Desktop\smash\clean_artworks'
output_artworks_folder = r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\23-coded_faces'

mask_image = [cv2.imread(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\4-lose_mask.png'), 
            cv2.imread(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\4-win_mask.png'),
            cv2.imread(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\4-lose_mask.png'),
            cv2.imread(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\4-lose_mask.png')]

if(MANUAL):
    # MANUAL
    try:
        character = "steve"
        artwork_reference = "0"
        final_width = 545
        final_height = 681
        final_x = 179
        final_y = 42

        if(os.path.isdir(os.path.join(faces_folder, character)) == False):
            print(character + ", can't be found. Skipped")
            raise Exception
        images = os.listdir(os.path.join(faces_folder, character))
        for face_image_file in images:
            if(face_image_file[0] != artwork_reference):
                continue
            print(f'{character}/{face_image_file}', end='')
            face_image = cv2.imread(os.path.join(faces_folder, character, face_image_file))

            for c in face_image_file[:-4]:
                artwork_image = cv2.cvtColor(cv2.imread(os.path.join(artworks_folder, character, c + ".png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
                artwork_image = deshadow(artwork_image)
                x_offset = int((27 * artwork_image.shape[1]) / final_width)
                y_offset = int((20 * artwork_image.shape[0]) / final_height)
                artwork_image = overlayImages(artwork_image, shadowOf(artwork_image, x_offset, y_offset))
                try:
                    os.mkdir(os.path.join(output_artworks_folder, character))
                except Exception:
                    pass
                cv2.imwrite(os.path.join(output_artworks_folder, character, c + ".png"), cv2.resize(artwork_image, (final_width, final_height))[final_y : final_y + face_image.shape[0], final_x : final_x + face_image.shape[1]])
    except Exception:
        pass

else:
    # AUTOMATIC
    queue = []
    for character_index in range(len(characters)):
        character = characters[character_index][1]
        if(os.path.isdir(os.path.join(faces_folder, character)) == False):
            print(character + ", can't be found. Skipped")
            continue
        images = os.listdir(os.path.join(faces_folder, character))
        for face_image_file in images:
            if(os.path.isfile(os.path.join(output_artworks_folder, character, face_image_file[0] + ".png"))):
                print(character + '/' + face_image_file + ", already done. Skipped")
            else:
                queue.append([character, face_image_file])
    print(queue)
    total_time = time.time()
    for image_index in range(len(queue)):
        image_time = time.time()
        character = queue[image_index][0]
        image_file = queue[image_index][1]
        image = cv2.imread(os.path.join(faces_folder, character, image_file))
        artwork_image = cv2.cvtColor(cv2.imread(os.path.join(artworks_folder, character, image_file[0] + ".png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
        player_sample_pixel = image[-3:-10]
        player = numpy.argmin([norm(player_sample_pixel - reference_pixel) for reference_pixel in SAMPLE_G])
        artwork_image = fun.addBackground(artwork_image, res.BACKGROUND_CLOSEUPS_BGRS[player])

        scale_range = numpy.linspace(0.707, 1.414, WIDTH_SAMPLES)
        width_range = (artwork_image.shape[1] * numpy.multiply(scale_range,scale_range)).astype(int)
        min_value = float("inf")
        approx_index = 0
        last_feedback_length = 0
        for i in range(len(width_range)):
            feedback = f'{character}/{image_file}, first scan: {i + 1} / {WIDTH_SAMPLES}'
            print(' ' * last_feedback_length, end='\r')
            print(feedback, end='\r')
            last_feedback_length = len(feedback)

            width = width_range[i]
            height = int((width * artwork_image.shape[0]) / artwork_image.shape[1])
            scaled_artwork = cv2.resize(artwork_image, (width, height))
            template_map = cv2.matchTemplate(scaled_artwork, image, cv2.TM_SQDIFF, None, mask_image[player])

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
            feedback = f'{character}/{image_file}, second scan:  {width - width_range[approx_index - 1] + 1} / {width_range[approx_index + 1] - width_range[approx_index - 1]}'
            print(' ' * last_feedback_length, end='\r')
            print(feedback, end='\r')
            last_feedback_length = len(feedback)

            height = int((width * artwork_image.shape[0]) / artwork_image.shape[1])
            scaled_artwork = cv2.resize(artwork_image, (width, height))
            template_map = cv2.matchTemplate(scaled_artwork, image, cv2.TM_SQDIFF, None, mask_image[player])

            curr_min_index = numpy.argmin(template_map)
            curr_min_x = curr_min_index % template_map.shape[1]
            curr_min_y = int((curr_min_index - curr_min_x) / template_map.shape[1])
            if(template_map[curr_min_y,curr_min_x] < min_value):
                final_width = width
                final_height = height
                final_x = curr_min_x
                final_y = curr_min_y
                min_value = template_map[curr_min_y,curr_min_x]

        feedback = f'{character}/{image_file}, saving cropped images...'
        print(' ' * last_feedback_length, end='\r')
        print(feedback, end='\r')
        last_feedback_length = len(feedback)
        for c in image_file[:-4]:
            artwork_image = cv2.cvtColor(cv2.imread(os.path.join(artworks_folder, character, c + ".png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
            artwork_image = deshadow(artwork_image)
            x_offset = int((27 * artwork_image.shape[1]) / final_width)
            y_offset = int((20 * artwork_image.shape[0]) / final_height)
            artwork_image = overlayImages(artwork_image, shadowOf(artwork_image, x_offset, y_offset))
            try:
                os.mkdir(os.path.join(output_artworks_folder, character))
            except Exception:
                pass
            cv2.imwrite(os.path.join(output_artworks_folder, character, c + ".png"), cv2.resize(artwork_image, (final_width, final_height))[final_y : final_y + image.shape[0], final_x : final_x + image.shape[1]])
        print(' ' * last_feedback_length, end='\r')
        print(f'{character}/{image_file}, done;  elapsed time: {formatTime(time.time() - image_time)}, ETL: {formatTime(((time.time() - total_time) * (len(queue) - image_index-1)) / (image_index+1))}')
    print(f'\ntotal elapsed time: {formatTime(time.time() - total_time)}')