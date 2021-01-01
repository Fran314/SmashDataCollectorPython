import time
import numpy
import sys
import os
from cv2 import cv2
import functions as fun
import resources as res
import customizable as custom

#--- INITIALIZATION ---#
t = time.time()

if(custom.LIVES > 8):
    print("The number of lives can't be greater than 8.")
    input("Press Enter to continue...")
    sys.exit(0)

#--- FIND MATCHES ---#
dirs = fun.mergeSort(os.listdir(custom.DATA_PATH))
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
    first_data = cv2.imread(os.path.join(custom.DATA_PATH, dirs[2*match_index]))
    second_data = cv2.imread(os.path.join(custom.DATA_PATH, dirs[2*match_index+1]))

    try:
        #--- FIRST IMAGE ---#
        types = []
        places = []
        character_indices = []
        times = []
        
        #--- GET NUMBER OF PLAYERS ---#
        players_colour = first_data[res.PLAYER_PIXEL]
        players_amount = numpy.argmin([numpy.linalg.norm(players_colour - colour) for colour in res.PLAYER_COLOURS])+2

        for curr_player_index in range(players_amount):
            #--- GET PLAYER TYPE ---#
            type_colour = first_data[res.TIME_PY, res.RIGHT_EDGE[players_amount-2][curr_player_index] + int(res.TIME_SEC_RD[players_amount-2] / 2)]
            curr_player_type = numpy.argmin([numpy.linalg.norm(type_colour - colour) for colour in res.PLAYER_TYPE_COLOURS])
            types.append(curr_player_type)

            #--- GET PLAYER PLACE ---#
            place_colour = first_data[res.PLACE_PIXEL_PY, res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.PLACE_PIXEL_RD[players_amount-2]]
            curr_place = numpy.argmin([numpy.linalg.norm(place_colour - colour) for colour in res.PLACE_COLOURS])+1
            places.append(curr_place)

            #--- GET PLAYER CHARACTER ---#
            character_closeup = first_data[138 : 138 + res.CHARACTER_CLOSEUP_HEIGHT[players_amount-2], res.LEFT_EDGE[players_amount-2][curr_player_index] : res.RIGHT_EDGE[players_amount-2][curr_player_index]].copy()
            curr_character_index = fun.getClosestCharacterByCloseup(character_closeup, res.CHARACTER_CLOSEUPS[players_amount-2], res.CHARACTER_CLOSEUPS_MASKS[players_amount-2][1 if curr_place == 1 else 0], res.BACKGROUND_CLOSEUPS_BGRS[curr_player_type])
            character_indices.append(curr_character_index)

            #--- GET PLAYER TIME ---#
            curr_time = ""
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.TIME_SEC_RD[players_amount-2]
            digit_y = res.TIME_PY[players_amount-2]
            digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
            if(digit == 10):
                error_message = f'unable to read P{curr_player_index+1}\'s time - can\'t find first digit'
                raise fun.InvalidData
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

                digit_x += res.TIME_DEC_SEP
                digit_image = fun.submat(first_data, digit_y, digit_x, res.BIG_DIGIT_HEIGHT, res.BIG_DIGIT_WIDTH)
                digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
                digit = fun.getClosestDigit(digit_image, res.BIG_DIGIT_IMAGES)
                if(digit == -1):
                    fun.showImage(digit_image)
                    error_message = f'unable to read P{curr_player_index+1}\'s time - error on last digit'
                    raise fun.InvalidData
                if(digit != 10):
                    curr_time = str(digit) + curr_time

                #TODO add support for 10:00 and up to 99:59
            times.append(curr_time)
        
        #--- FIRST DATA - ERROR CHECKING ---#
        if(fun.isValidFirstData(places, times) == False):
            error_message = f'positions ({places}) and times ({times}) are invalid'
            raise fun.InvalidData


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
                    if(numpy.sum(fun.polarizeImage(fun.submat(second_data, res.AP_STENCILS_TOPY[players_amount - 2] + j, res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD, res.AP_STENCIL_HEIGHT, res.SMALL_DIGIT_WIDTH), 170)) == 0 and
                        numpy.sum(fun.polarizeImage(fun.submat(second_data, res.AP_STENCILS_TOPY[players_amount - 2] + j + res.DARK_BAND_SEP, res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD, res.AP_STENCIL_HEIGHT, res.SMALL_DIGIT_WIDTH), 170)) > 0):
                        anchor_point = res.AP_STENCILS_TOPY[players_amount - 2] + j
            if(anchor_point == 0):
                error_message = f'couldn\'t find anchor point for P{curr_player_index+1}'
                raise fun.InvalidData

            #--- GET PLAYER FALLS ---#
            curr_fall_list = []
            fall_icon_px = res.LEFT_EDGE[players_amount-2][curr_player_index] + res.FALL_ICON_LD[players_amount-2]
            fall_icon_py = anchor_point + res.FALL_ICON_YO
            for j in range(custom.LIVES):
                fall_image = fun.submat(second_data, fall_icon_py, int(fall_icon_px), res.FALL_ICON_SIZE[players_amount-2], res.FALL_ICON_SIZE[players_amount-2])
                fall_image = cv2.resize(fall_image, (50,50))
                killer = fun.getClosestPlayerByIcon(fall_image, [res.CHARACTER_ICONS[index] for index in character_indices], res.BACKGROUND_ICONS_BGRS[types[curr_player_index]])
                if(killer == -1):
                    break
                if(killer == curr_player_index):
                    error_message = f'P{curr_player_index+1}\'s first killer read as itself'
                    raise fun.InvalidData
                curr_fall_list.append(killer + 1)
                fall_icon_px += res.FALL_ICON_SEP

            #--- GET PLAYER SELFDESTRUCTS ---#
            digit_y = anchor_point + res.SELFDESTR_YO
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.SELFDESTR_RD
            digit_image = fun.submat(second_data, digit_y, digit_x, res.SMALL_DIGIT_HEIGHT, res.SMALL_DIGIT_WIDTH)
            digit_image = fun.polarizeImage(digit_image, res.POLARIZATION_THRESHOLD)
            curr_selfdestruct = fun.getClosestDigit(digit_image, res.SMALL_DIGIT_IMAGES)
            for j in range(curr_selfdestruct):
                curr_fall_list.append(curr_player_index+1)

            falls.append(curr_fall_list)

            #--- FALLS AND SELFDESTRUCTS - ERROR CHECKING ---#
            if(places[curr_player_index] != 1 and len(curr_fall_list) != custom.LIVES):
                error_message = f'G{curr_player_index+1} (not in first place) died a number of times different from {custom.LIVES} ({curr_fall_list})'
                raise fun.InvalidData
            if(places[curr_player_index] == 1 and len(curr_fall_list) >= custom.LIVES):
                error_message = f'G{curr_player_index+1} (in first place) died more than {custom.LIVES-1} times ({curr_fall_list})'
                raise fun.InvalidData


            #--- GET PLAYER GIVEN DAMAGE ---#
            given_damage = 0
            digit_y = anchor_point + res.GVN_DMG_YO
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.DMG_RD
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
            digit_x = res.RIGHT_EDGE[players_amount-2][curr_player_index] + res.DMG_RD
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
        for curr_player_index in range(players_amount):
            taken_given_dmg_difference += taken_damages[curr_player_index] - given_damages[curr_player_index]
        if(taken_given_dmg_difference < 0 or taken_given_dmg_difference >= custom.TAKEN_GIVEN_DMG_THRESHOLD):
            error_message = f'too big of a difference between total taken damage and total given damage ({taken_given_dmg_difference})'
            raise fun.InvalidData


        #--- CONVERT MATCH DATA TO A STRING ---#
        output_strings.append(fun.convertMatchToString(dirs[2*match_index], [res.CHARACTER_INFOS[index][0] for index in character_indices], places, times, falls, given_damages, taken_damages))

    except fun.InvalidData:
        problematic_matches.append([match_index, error_message])
        print(f'Problem: {error_message}')
        output_strings.append("")


if(len(problematic_matches) > 0):
    #--- HALFWAY CHECKUP ---#
    print(f'elapsed time: {(time.time() - t):.3f} s')

    #--- WRITE TEMPORARY OUTPUT ---#
    output_file = open(custom.OUTPUT_PATH, 'w')
    for match in output_strings[:-1]:
        output_file.write(match)
        output_file.write("\n")
    output_file.write(output_strings[-1])
    output_file.close()

    #--- HANDLE PROBLEMATIC MATCHES ---#
    print(f'Unable to read data of {len(problematic_matches)} matches out of {tot_matches}. Please enter the data manually.')
    match_counter = 0
    try:
        for problematic_match in problematic_matches:
            print(f'Problematic match #{match_counter+1} of {len(problematic_matches)} (match #{problematic_match[0]+1})')
            print(f'Problem: {problematic_match[1]}')

            first_data = cv2.imread(os.path.join(custom.DATA_PATH, dirs[2*problematic_match[0]]))
            second_data = cv2.imread(os.path.join(custom.DATA_PATH, dirs[2*problematic_match[0]+1]))
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

                    regex = f'[2-{res.MAX_PLAYERS}]'
                    players_amount = int(fun.readInput(f'Enter number of players: ', regex))
                    for curr_player_index in range(players_amount):
                        #--- GET PLAYER CHARACTER ---#
                        regex = f'({")|(".join([name[0] for name in res.CHARACTER_INFOS])})'
                        player_character = fun.readInput(f'Enter P{curr_player_index+1} character: ', regex)
                        for name in res.CHARACTER_INFOS:
                            if(name[0] == player_character.upper()):
                                player_character = name[1]
                                break
                        character_indices.append(player_character)

                        #--- GET PLAYER POSITION ---#
                        regex = f'[1-{players_amount}]'
                        player_position = fun.readInput(f'Enter P{curr_player_index+1} position: ', regex)
                        positions.append(int(player_position))

                        #--- GET PLAYER TIME ---#
                        regex = "([1-9]?[0-9]:[0-5][0-9])?"
                        player_time = fun.readInput(f'Enter P{curr_player_index+1} time (m:ss): ', regex)
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
                    for curr_player_index in range(players_amount):
                        #--- GET PLAYER FALLS ---#
                        regex = "([" + "".join(map(str, range(1, curr_player_index+1))) + "".join(map(str, range(curr_player_index+2, players_amount+1))) + "](,[" + "".join(map(str, range(1, curr_player_index+1))) + "".join(map(str, range(curr_player_index+2, players_amount+1))) + "]){0," + str(custom.LIVES - 1) + "})?"
                        player_falls_string = fun.readInput(f'Enter P{curr_player_index+1} falls (separated by commas): ', regex)
                        fall_list = []
                        for j in range(len(player_falls_string)):
                            if(j % 2 == 0):
                                fall_list.append(int(player_falls_string[j]))
                        falls.append(fall_list)

                        #--- GET PLAYER SELFDESTRUCTS --#
                        regex = f'[0-{custom.LIVES}]'
                        player_selfdestruct_string = fun.readInput(f'Enter P{curr_player_index+1} selfdestructs: ', regex)
                        for j in range(int(player_selfdestruct_string)):
                            fall_list.append(curr_player_index+1)

                        #--- ERROR CHECKING ---#
                        if(positions[curr_player_index] != 1 and len(fall_list) != custom.LIVES):
                            error_message = f'P{curr_player_index+1} (not in first position) died a number of times different from {custom.LIVES} ({fall_list})'
                            raise fun.InvalidData
                        if(positions[curr_player_index] == 1 and len(fall_list) >= custom.LIVES):
                            error_message = f'P{curr_player_index+1} (in first position) died more than {custom.LIVES-1} times ({fall_list})'
                            raise fun.InvalidData
                        
                        #--- GET PLAYER GIVEN DAMAGE ---#
                        regex = "[0-9]+"
                        player_given_damage = fun.readInput(f'Enter P{curr_player_index+1} given damage (without percent): ', regex)
                        given_damages.append(int(player_given_damage))

                        #--- GET PLAYER TAKEN DAMAGE ---#
                        regex = "[0-9]+"
                        player_taken_damage = fun.readInput(f'Enter P{curr_player_index+1} taken damage (without percent): ', regex)
                        taken_damages.append(int(player_taken_damage))
                        
                    #--- SECOND DATA - ERROR CHECKING ---#
                    taken_given_dmg_difference = 0
                    for curr_player_index in range(players_amount):
                        taken_given_dmg_difference += taken_damages[curr_player_index] - given_damages[curr_player_index]
                    if(taken_given_dmg_difference < 0 or taken_given_dmg_difference >= custom.TAKEN_GIVEN_DMG_THRESHOLD):
                        error_message = f'too big of a difference between total taken damage and total given damage ({taken_given_dmg_difference})'
                        raise fun.InvalidData


                    #--- CONVERT MATCH DATA TO A STRING ---#
                    output_strings[problematic_match[0]] = fun.convertMatchToString(dirs[2*problematic_match[0]], character_indices, positions, times, falls, given_damages, taken_damages)

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
output_file = open(custom.OUTPUT_PATH, 'w')
for match in output_strings[:-1]:
    output_file.write(match)
    output_file.write("\n")
output_file.write(output_strings[-1])
output_file.close()


#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')
input("Press Enter to continue...")