import time
import numpy
import pytesseract
from cv2 import cv2
import re
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# pylint: disable=import-error
import functions as fun
import resources as res

#--- INITIALIZATION ---#
t = time.time()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


matches = fun.mergeSort(os.listdir(r'C:\Users\franc\Desktop\smash\reference_characters'))
CHARACTER_NAMES = fun.readTSV(r'C:\Users\franc\Desktop\characters_info.tsv')

for match in matches:
    first_data = cv2.imread(os.path.join(r'C:\Users\franc\Desktop\smash\reference_characters', match))

    places = []
    characters = []
    
    players = 3

    for i in range(players):
        #--- GET PLAYER CHARACTER ---#
        if(i != 2):
            curr_name_rect = fun.submat(first_data, res.NAME_PY[players-2], res.LEFT_EDGE[players-2][i] + res.NAME_LD, res.NAME_HEIGHT, res.NAME_WIDTH)
        else:
            curr_name_rect = fun.submat(first_data, res.NAME_PY[players-2] + res.WIN_OFFSET, res.LEFT_EDGE[players-2][i] + res.NAME_LD, res.NAME_HEIGHT, res.NAME_WIDTH)
        curr_name_rect = fun.polarizeImage(curr_name_rect, res.POLARIZATION_THRESHOLD)
        curr_name = fun.normalizeName(pytesseract.image_to_string(curr_name_rect))
        character_name_ed = []
        for j in range(len(CHARACTER_NAMES)):
            character_name_ed.append(fun.editDistance(curr_name, CHARACTER_NAMES[j][0].upper()))
        min_index = numpy.argmin(character_name_ed)
        name = ""
        character_image = first_data[138 : 138 + 255, res.LEFT_EDGE[players-2][i] : res.RIGHT_EDGE[players-2][i]]
        if(character_name_ed[min_index] >= len(CHARACTER_NAMES[min_index][0]) / 3):
            print(f'G{i+1}\'s name is too uncertain')

            fun.showImage(character_image, "boh", 20)

            regex = f'({")|(".join([name[0].upper() for name in CHARACTER_NAMES])})'
            player_name = fun.readInput(f'Enter G{i+1} character: ', regex)
            for name in CHARACTER_NAMES:
                if(name[0].upper() == player_name.upper()):
                    player_name = name[2]
                    break
        else:
            player_name = CHARACTER_NAMES[min_index][2]
            
        folder_path = os.path.join(r'C:\Users\franc\Desktop\smash\faces', player_name)
        try:
            os.mkdir(folder_path)
        except Exception:
            pass
        already_existing = len(os.listdir(folder_path))
        cv2.imwrite(os.path.join(folder_path, str(already_existing) + ".png"), character_image)

# for name in CHARACTER_NAMES:
#     print(name[1])
#     if(os.path.isdir(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1])) == False):
#         continue
#     files = os.listdir(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1]))
#     if(len(files) == 1):
#         cv2.imwrite(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "01234567.png"), cv2.imread(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "0.png")))
#         os.remove(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "0.png"))
#         pass
#     elif(len(files) == 2):
#         cv2.imwrite(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "0246.png"), cv2.imread(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "0.png")))
#         cv2.imwrite(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "1357.png"), cv2.imread(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "1.png")))
#         os.remove(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "0.png"))
#         os.remove(os.path.join(r'C:\Users\franc\Desktop\smash\faces', name[1], "1.png"))
#         pass


#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')