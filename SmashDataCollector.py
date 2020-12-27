import time
import numpy
import sys
import os
import pytesseract
from cv2 import cv2
import functions as fun
import resources as res
import customizable as custom

#--- INITIALIZATION ---#
t = time.time()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#--- FIND MATCHES ---#
dirs = fun.mergeSort(os.listdir(custom.data_path))
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
    first_data = cv2.imread(os.path.join(custom.data_path, dirs[2*match_index]))
    second_data = cv2.imread(os.path.join(custom.data_path, dirs[2*match_index+1]))

    try:
        #--- FIRST IMAGE ---#
        places = []
        character_indices = []
        times = []
        
        #--- GET NUMBER OF PLAYERS ---#
        # TODO: 2-3-4 players only
        players_colour = first_data[res.PLAYER_PIXEL]
        players = numpy.argmin([numpy.linalg.norm(players_colour - colour) for colour in res.PLAYER_COLOURS])+2

        for player_index in range(players):
            #--- GET PLAYER PLACE ---#
            curr_place_colour = first_data[res.PLACE_PIXEL_PY, res.RIGHT_EDGE[players-2][player_index] + res.PLACE_PIXEL_RD[players-2]]
            curr_place = numpy.argmin([numpy.linalg.norm(curr_place_colour - place_col) for place_col in res.PLACE_COLOURS])+1
            places.append(curr_place)

            #--- GET PLAYER CHARACTER ---#
            curr_character_image = first_data[138 : 138 + 255, res.LEFT_EDGE[players-2][player_index] : res.RIGHT_EDGE[players-2][player_index]].copy()
            curr_character_index = fun.getClosestCharacterByCloseup(curr_character_image, res.CHARACTER_CLOSEUPS[players-2], res.CHARACTER_CLOSEUPS_MASKS[players-2][curr_place == 1], res.BACKGROUND_BGRS[player_index])
            character_indices.append(curr_character_index)

            #--- GET PLAYER TIME ---#
            curr_time = ""
            digit_x = res.RIGHT_EDGE[players-2][player_index] + res.TIME_SEC_RD[players-2]
            digit_y = res.TIME_PY[players-2]
            digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
            if(digit != -1):
                curr_time = str(digit)

                digit_x += res.TIME_DEC_SEP
                digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
                curr_time = str(digit) + curr_time

                digit_x += res.TIME_MIN_SEP
                digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
                curr_time = str(digit) + ":" + curr_time
            times.append(curr_time)

            #--- PLAYER - ERROR CHECKING ---#
            if(places[player_index] < 1 or places[player_index] > players):
                error_message = f'G{player_index+1}\'s position ({places[player_index]}) is invalid'
                raise fun.InvalidData
            if(fun.isValidTime(times[player_index]) == False):
                error_message = f'G{player_index+1}\'s time ({times[player_index]}) is invalid'
                raise fun.InvalidData
        
        #--- FIRST DATA - ERROR CHECKING ---#
        if(fun.isValidFirstData(places, times) == False):
            error_message = f'positions ({places}) and times ({times}) are invalid'
            #raise InvalidData


        #--- SECOND IMAGE ---#
        falls = []
        given_damages = []
        taken_damages = []
        for player_index in range(players):
            #--- ESTABLISH ANCHOR POINTS ---#
            stencil_height = res.ANCHOR_POINT_STENCILS[player_index].shape[0]
            stencil_width = res.ANCHOR_POINT_STENCILS[player_index].shape[1]
            data_height = second_data.shape[0]
            min_norm = fun.imageDistance(res.ANCHOR_POINT_STENCILS[player_index], fun.submat(second_data, 0, res.LEFT_EDGE[players-2][player_index] + res.ANCHOR_POINT_LD[players-2], stencil_height, stencil_width))
            pos_min = 0
            for j in range(data_height - stencil_height):
                curr_norm = fun.imageDistance(res.ANCHOR_POINT_STENCILS[player_index], fun.submat(second_data, j, res.LEFT_EDGE[players-2][player_index] + res.ANCHOR_POINT_LD[players-2], stencil_height, stencil_width))
                if(curr_norm < min_norm):
                    min_norm = curr_norm
                    pos_min = j
            anchor_point = pos_min

            #--- GET PLAYER FALLS ---#
            curr_fall_list = []
            fall_icon_px = res.LEFT_EDGE[players-2][player_index] + res.FALL_ICON_LD[players-2]
            fall_icon_py = anchor_point + res.FALL_ICON_YO
            for j in range(custom.LIVES):
                fall_image = fun.submat(second_data, fall_icon_py, int(fall_icon_px), res.FALL_ICON_SIZE[players-2], res.FALL_ICON_SIZE[players-2])
                fall_image = cv2.resize(fall_image, (50,50))
                killer = fun.getClosestCharacterByIcon(fall_image, res.EMPTY_ICONS[player_index], character_indices)
                if(killer == -1):
                    break
                if(killer == player_index):
                    error_message = f'G{player_index+1}\'s first killer read as itself'
                    raise fun.InvalidData
                curr_fall_list.append(killer + 1)
                fall_icon_px += res.FALL_ICON_SEP

            #--- GET PLAYER SELFDESTRUCTS ---#
            digit_y = anchor_point + res.SELFDESTR_YO
            digit_x = res.RIGHT_EDGE[players-2][player_index] + res.SELFDESTR_RD
            digit_image = fun.submat(second_data, digit_y, digit_x, res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            curr_selfdestruct = fun.getClosestDigit(digit_image, res.SMALL_DIGIT_IMAGES)
            for j in range(curr_selfdestruct):
                curr_fall_list.append(player_index+1)

            falls.append(curr_fall_list)

            #--- FALLS AND SELFDESTRUCTS - ERROR CHECKING ---#
            if(places[player_index] != 1 and len(curr_fall_list) != custom.LIVES):
                error_message = f'G{player_index+1} (not in first place) died a number of times different from {custom.LIVES} ({curr_fall_list})'
                raise fun.InvalidData
            if(places[player_index] == 1 and len(curr_fall_list) >= custom.LIVES):
                error_message = f'G{player_index+1} (in first place) died more than {custom.LIVES-1} times ({curr_fall_list})'
                raise fun.InvalidData


            #--- GET PLAYER GIVEN DAMAGE ---#
            given_damage = 0
            digit_y = anchor_point + res.GVN_DMG_YO
            digit_x = res.RIGHT_EDGE[players-2][player_index] + res.DMG_RD
            digit = 0
            decimal_place = 1
            while(digit != -1):
                digit_image = fun.submat(second_data, digit_y, int(digit_x), res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getClosestDigit(digit_image, res.SMALL_DIGIT_IMAGES)
                if(digit != -1):
                    given_damage += digit*decimal_place
                digit_x -= res.SMALL_DIGIT_SEP
                decimal_place *= 10
            given_damages.append(given_damage)
            
            #--- GET PLAYER TAKEN DAMAGE ---#
            taken_damage = 0
            digit_y = anchor_point + res.TKN_DMG_YO
            digit_x = res.RIGHT_EDGE[players-2][player_index] + res.DMG_RD
            digit = 0
            decimal_place = 1
            while(digit != -1):
                digit_image = fun.submat(second_data, digit_y, int(digit_x), res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getClosestDigit(digit_image, res.SMALL_DIGIT_IMAGES)
                if(digit != -1):
                    taken_damage += digit*decimal_place
                digit_x -= res.SMALL_DIGIT_SEP
                decimal_place *= 10
            taken_damages.append(taken_damage)
        
        #--- SECOND DATA - ERROR CHECKING ---#
        taken_given_dmg_difference = 0
        for player_index in range(players):
            taken_given_dmg_difference += taken_damages[player_index] - given_damages[player_index]
        if(taken_given_dmg_difference < 0 or taken_given_dmg_difference >= custom.TAKEN_GIVEN_DMG_THRESHOLD):
            error_message = f'too big of a difference between total taken damage and total given damage ({taken_given_dmg_difference})'
            raise fun.InvalidData


        #--- CONVERT MATCH DATA TO A STRING ---#
        output_strings.append(fun.convertMatchToString(dirs[2*match_index], players, character_indices, places, times, falls, given_damages, taken_damages))

    except fun.InvalidData:
        problematic_matches.append([match_index, error_message])
        print(f'Problem: {error_message}')
        output_strings.append("")


#--- HALFWAY CHECKUP ---#
if(len(problematic_matches) > 0):
    print(f'elapsed time: {(time.time() - t):.3f} s')

    
#--- WRITE OUTPUT ---#
output_file = open(os.path.join(custom.output_path, "output.tsv"), 'w')
for match in output_strings[:-1]:
    output_file.write(match)
    output_file.write("\n")
output_file.write(output_strings[-1])
output_file.close()


#--- HANDLE PROBLEMATIC MATCHES ---#
if(len(problematic_matches) > 0):
    print(f'Unable to read data of {len(problematic_matches)} matches out of {tot_matches}. Please enter the data manually.')
match_counter = 0
try:
    for problematic_match in problematic_matches:
        print(f'Problematic match #{match_counter+1} of {len(problematic_matches)} (match #{problematic_match[0]+1})')
        print(f'Problem: {problematic_match[1]}')

        first_data = cv2.imread(os.path.join(custom.data_path, dirs[2*problematic_match[0]]))
        second_data = cv2.imread(os.path.join(custom.data_path, dirs[2*problematic_match[0]+1]))
        valid_data = False
        while(valid_data == False):
            valid_data = True
            print('At any time, you can enter "SKIP" to skip the current match or enter "SKIP ALL" to skip all the remaining matches.')
            try:
                #--- FIRST IMAGE ---#
                cv2.destroyAllWindows()
                fun.showImage(cv2.resize(first_data, (640, 360)), f'match {problematic_match[0]+1}/{int(len(dirs)/2)}, first screenshot', 20)
                character_indices = []
                positions = []
                times = []

                regex = f'[2-8]'
                players = int(fun.readInput(f'Enter number of players: ', regex))
                for player_index in range(players):
                    #--- GET PLAYER CHARACTER ---#
                    regex = f'({")|(".join([name[0] for name in res.CHARACTER_INFOS])})'
                    player_character = fun.readInput(f'Enter G{player_index+1} character: ', regex)
                    for name in res.CHARACTER_INFOS:
                        if(name[0] == player_character.upper()):
                            player_character = name[1]
                            break
                    character_indices.append(player_character)

                    #--- GET PLAYER POSITION ---#
                    regex = f'[1-{players}]'
                    player_position = fun.readInput(f'Enter G{player_index+1} position: ', regex)
                    positions.append(int(player_position))

                    #--- GET PLAYER TIME ---#
                    regex = "([0-9]:[0-5][0-9])?"
                    player_time = fun.readInput(f'Enter G{player_index+1} time (m:ss): ', regex)
                    times.append(player_time)
                
                #--- FIRST DATA - ERROR CHECKING ---#
                if(fun.isValidFirstData(positions, times) == False):
                    error_message = f'positions ({positions}) and times ({times}) are invalid'
                    raise fun.InvalidData


                #--- SECOND IMAGE ---#
                cv2.destroyAllWindows()
                fun.showImage(cv2.resize(second_data, (640, 360)), f'match {problematic_match[0]+1}/{int(len(dirs)/2)}, second screenshot', 20)
                falls = []
                given_damages = []
                taken_damages = []
                for player_index in range(players):
                    #--- GET PLAYER FALLS ---#
                    regex = "([" + "".join(map(str, range(1, player_index+1))) + "".join(map(str, range(player_index+2, players+1))) + "](,[" + "".join(map(str, range(1, player_index+1))) + "".join(map(str, range(player_index+2, players+1))) + "]){0," + str(custom.LIVES - 1) + "})?"
                    player_falls_string = fun.readInput(f'Enter G{player_index+1} falls (separated by commas): ', regex)
                    fall_list = []
                    for j in range(len(player_falls_string)):
                        if(j % 2 == 0):
                            fall_list.append(int(player_falls_string[j]))
                    falls.append(fall_list)

                    #--- GET PLAYER SELFDESTRUCTS --#
                    regex = f'[0-{custom.LIVES}]'
                    player_selfdestruct_string = fun.readInput(f'Enter G{player_index+1} selfdestructs: ', regex)
                    for j in range(int(player_selfdestruct_string)):
                        fall_list.append(player_index+1)

                    #--- ERROR CHECKING ---#
                    if(positions[player_index] != 1 and len(fall_list) != custom.LIVES):
                        error_message = f'G{player_index+1} (not in first position) died a number of times different from {custom.LIVES} ({fall_list})'
                        raise fun.InvalidData
                    if(positions[player_index] == 1 and len(fall_list) >= custom.LIVES):
                        error_message = f'G{player_index+1} (in first position) died more than {custom.LIVES-1} times ({fall_list})'
                        raise fun.InvalidData
                    
                    #--- GET PLAYER GIVEN DAMAGE ---#
                    regex = "[0-9]+"
                    player_given_damage = fun.readInput(f'Enter G{player_index+1} given damage (without percent): ', regex)
                    given_damages.append(int(player_given_damage))

                    #--- GET PLAYER TAKEN DAMAGE ---#
                    regex = "[0-9]+"
                    player_taken_damage = fun.readInput(f'Enter G{player_index+1} taken damage (without percent): ', regex)
                    taken_damages.append(int(player_taken_damage))
                    
                #--- SECOND DATA - ERROR CHECKING ---#
                taken_given_dmg_difference = 0
                for player_index in range(players):
                    taken_given_dmg_difference += taken_damages[player_index] - given_damages[player_index]
                if(taken_given_dmg_difference < 0 or taken_given_dmg_difference >= custom.TAKEN_GIVEN_DMG_THRESHOLD):
                    error_message = f'too big of a difference between total taken damage and total given damage ({taken_given_dmg_difference})'
                    raise fun.InvalidData


                #--- CONVERT MATCH DATA TO A STRING ---#
                output_strings[problematic_match[0]] = fun.convertMatchToString(dirs[2*problematic_match[0]], players, character_indices, positions, times, falls, given_damages, taken_damages)

            except fun.InvalidData:
                print("The data inserted has an error or is self-contradictory in some way.")
                print("Problem: " + error_message)
                print("Please check and insert the data again. If the data doesn't have any error and isn't actually self-contradicotry, please skip and add the data manually to the output.")
                valid_data = False
            except fun.Skip:
                valid_data = True
        
        cv2.destroyAllWindows()
        print("The data inserted is valid!")
        match_counter += 1
except fun.SkipAll:
    pass

#--- WRITE OUTPUT ---#
output_file = open(os.path.join(custom.output_path, "output.tsv"), 'w')
for match in output_strings[:-1]:
    output_file.write(match)
    output_file.write("\n")
output_file.write(output_strings[-1])
output_file.close()


#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
input("Press Enter to continue...")