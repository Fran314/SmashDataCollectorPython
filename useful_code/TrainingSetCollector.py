import time
import numpy
import sys
import os
from cv2 import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# pylint: disable=import-error
import functions as fun
import resources as res
import customizable as custom

from random import randrange

TRAINING_SET_IMAGES_PATH = r'C:\Users\franc\Desktop\foto'
TRAINING_SET_OUTPUT_PATH = r'C:\Users\franc\Desktop\training_set'

#--- INITIALIZATION ---#
t = time.time()

#--- FIND MATCHES ---#
dirs = fun.mergeSort(os.listdir(TRAINING_SET_IMAGES_PATH))
if(len(dirs) % 2 != 0):
    print("Odd number of images present in data. Ignoring the last image.")
    dirs = dirs[:-1]
tot_matches = int(len(dirs)/2)

#--- ANALYZE MATCHES ---#
error_message = ""
output_strings = []
problematic_matches = []
for match_index in range(tot_matches):
    print(f'Match #{match_index+1} out of {tot_matches}')
    first_data = cv2.imread(os.path.join(TRAINING_SET_IMAGES_PATH, dirs[2*match_index]))
    second_data = cv2.imread(os.path.join(TRAINING_SET_IMAGES_PATH, dirs[2*match_index+1]))

    try:
        #--- FIRST IMAGE ---#
        types = []
        places = []
        character_indices = []
        times = []
        
        #--- GET NUMBER OF PLAYERS ---#
        players_amount = 0
        for i in range(3):
            if(all([numpy.linalg.norm(first_data[pixel]) < 20 for pixel in res.PLAYER_PIXELS[i]])):
                players_amount = i+2
                break
        if(players_amount == 0):
            error_message = f'unable to tell how many players'
            raise fun.InvalidData
        
        language_image = fun.submat(first_data, res.LANGUAGE_YO[players_amount-2], res.LEFT_EDGE[players_amount-2][0] + res.LANGUAGE_LD, res.LANGUAGE_HEIGHT, res.LANGUAGE_WIDTH[players_amount-2])
        language = fun.getClosestLanguageType(language_image, res.LANGUAGES[players_amount-2])


        for curr_player_index in range(players_amount):
            #--- GET PLAYER TYPE ---#
            type_colour = first_data[res.TIME_PY[players_amount-2], res.RIGHT_EDGE[players_amount-2][curr_player_index] + int(res.TIME_SEC_RD[players_amount-2] / 2)]
            curr_player_type = numpy.argmin([numpy.linalg.norm(type_colour - colour) for colour in res.PLAYER_TYPE_COLOURS])
            types.append(curr_player_type)


        #--- SECOND IMAGE ---#
        falls = []
        given_damages = []
        taken_damages = []
        for curr_player_index in range(players_amount):
            #--- ESTABLISH ANCHOR POINTS ---#
            ap_area = second_data[res.AP_STENCILS_TOPY[players_amount - 2] : res.AP_STENCILS_BOTY[players_amount - 2], 
                                res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.AP_STENCILS_RD : res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.AP_STENCILS_RD + res.AP_STENCIL_WIDTH].copy()
            ap_matches = cv2.matchTemplate(ap_area, res.ANCHOR_POINT_STENCILS[types[curr_player_index]], cv2.TM_CCOEFF)
            anchor_point = 0
            for j in range(ap_matches.shape[0]):
                upper_span = min(j, res.AP_SPAN_RANGE)
                lower_span = min(ap_matches.shape[0] - j - 1, res.AP_SPAN_RANGE)
                region = ap_matches[j - upper_span : j + lower_span]
                if(numpy.argmax(region) == upper_span):
                    if(numpy.sum(fun.polarizeImage(fun.submat(second_data, res.AP_STENCILS_TOPY[players_amount - 2] + j, res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD[language], res.AP_STENCIL_HEIGHT, res.SMALL_DIGIT_WIDTH), 170)) == 0 and
                        numpy.sum(fun.polarizeImage(fun.submat(second_data, res.AP_STENCILS_TOPY[players_amount - 2] + j + res.DARK_BAND_SEP, res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD[language], res.AP_STENCIL_HEIGHT, res.SMALL_DIGIT_WIDTH), 170)) > 0):
                        anchor_point = res.AP_STENCILS_TOPY[players_amount - 2] + j
            if(anchor_point == 0):
                error_message = f'couldn\'t find anchor point for P{curr_player_index+1}'
                raise fun.InvalidData

            #--- GET PLAYER SELFDESTRUCTS ---#
            digit_y = anchor_point + res.SELFDESTR_YO
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD[language]
            digit_image = fun.submat(second_data, digit_y, digit_x, res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            digit = fun.getClosestDigit(digit_image, res.SMALL_DIGIT_IMAGES)
            TEMP_OUTPUT = os.path.join(TRAINING_SET_OUTPUT_PATH, str(digit))
            FILE_NAME = str(randrange(100000000)) + ".png"
            cv2.imwrite(os.path.join(TEMP_OUTPUT, FILE_NAME), digit_image)

            #--- GET PLAYER GIVEN DAMAGE ---#
            given_damage = 0
            digit_y = anchor_point + res.GVN_DMG_YO
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.DMG_RD[language]
            digit = 0
            decimal_place = 1
            while(digit != -1):
                digit_image = fun.submat(second_data, digit_y, int(digit_x), res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getClosestDigit(digit_image, res.SMALL_DIGIT_IMAGES)
                if(digit != -1):
                    TEMP_OUTPUT = os.path.join(TRAINING_SET_OUTPUT_PATH, str(digit))
                    FILE_NAME = str(randrange(100000000)) + ".png"
                    cv2.imwrite(os.path.join(TEMP_OUTPUT, FILE_NAME), digit_image)
                    given_damage += digit*decimal_place
                digit_x -= res.SMALL_DIGIT_SEP
                decimal_place *= 10
            
            #--- GET PLAYER TAKEN DAMAGE ---#
            taken_damage = 0
            digit_y = anchor_point + res.TKN_DMG_YO
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.DMG_RD[language]
            digit = 0
            decimal_place = 1
            while(digit != -1):
                digit_image = fun.submat(second_data, digit_y, int(digit_x), res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getClosestDigit(digit_image, res.SMALL_DIGIT_IMAGES)
                if(digit != -1):
                    TEMP_OUTPUT = os.path.join(TRAINING_SET_OUTPUT_PATH, str(digit))
                    FILE_NAME = str(randrange(100000000)) + ".png"
                    cv2.imwrite(os.path.join(TEMP_OUTPUT, FILE_NAME), digit_image)
                    taken_damage += digit*decimal_place
                digit_x -= res.SMALL_DIGIT_SEP
                decimal_place *= 10

    except fun.InvalidData:
        problematic_matches.append([match_index, error_message])
        print(f'Problem: {error_message}')
        output_strings.append("")

#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
input("Press Enter to continue...")