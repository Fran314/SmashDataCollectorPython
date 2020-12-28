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

source_path = r'C:\Users\franc\Desktop\smash\reference_characters'
output_path = r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\4-faces'

CHARACTER_NAMES = fun.readTSV(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\characters_info.tsv')

# matches = fun.mergeSort(os.listdir(source_path))
# for match in matches:
#     first_data = cv2.imread(os.path.join(source_path, match))

#     places = []
#     characters = []
    
#     players = 4

#     for i in range(players):
#         #--- GET PLAYER CHARACTER ---#
#         if(i != 2):
#             curr_name_rect = fun.submat(first_data, res.NAME_PY[players-2], res.LEFT_EDGE[players-2][i] + res.NAME_LD, res.NAME_HEIGHT, res.NAME_WIDTH)
#         else:
#             curr_name_rect = fun.submat(first_data, res.NAME_PY[players-2] + res.WIN_OFFSET, res.LEFT_EDGE[players-2][i] + res.NAME_LD, res.NAME_HEIGHT, res.NAME_WIDTH)
#         curr_name_rect = fun.polarizeImage(curr_name_rect, res.POLARIZATION_THRESHOLD)
#         curr_name = fun.normalizeName(pytesseract.image_to_string(curr_name_rect))
#         character_name_ed = []
#         for j in range(len(CHARACTER_NAMES)):
#             character_name_ed.append(fun.editDistance(curr_name, CHARACTER_NAMES[j][0].upper()))
#         min_index = numpy.argmin(character_name_ed)
#         name = ""
#         #character_image = first_data[138 : 138 + 255, res.LEFT_EDGE[players-2][i] : res.RIGHT_EDGE[players-2][i]]
#         character_image = first_data[138 : 138 + 245, res.LEFT_EDGE[players-2][i] : res.RIGHT_EDGE[players-2][i]]
#         if(character_name_ed[min_index] >= len(CHARACTER_NAMES[min_index][0]) / 3):
#             print(f'G{i+1}\'s name is too uncertain')

#             fun.showImage(character_image, "boh", 20)

#             regex = f'({")|(".join([name[0].upper() for name in CHARACTER_NAMES])})'
#             player_name = fun.readInput(f'Enter G{i+1} character: ', regex)
#             for name in CHARACTER_NAMES:
#                 if(name[0].upper() == player_name.upper()):
#                     player_name = name[2]
#                     break
#         else:
#             player_name = CHARACTER_NAMES[min_index][2]
            
#         folder_path = os.path.join(r'C:\Users\franc\Desktop\smash\faces', player_name)
#         try:
#             os.mkdir(folder_path)
#         except Exception:
#             pass
#         already_existing = len(os.listdir(folder_path))
#         cv2.imwrite(os.path.join(folder_path, str(already_existing) + ".png"), character_image)

for name in CHARACTER_NAMES:
    print(name[1], end='')
    character_folder = os.path.join(output_path, name[1])
    if(os.path.isdir(character_folder) == False):
        print(f', not found.')
        continue
    files = os.listdir(character_folder)
    if(len(files) == 1):
        os.rename(os.path.join(character_folder, "0.png"), os.path.join(character_folder, "01234567.png"))
    elif(len(files) == 2):
        os.rename(os.path.join(character_folder, "0.png"), os.path.join(character_folder, "0246.png"))
        os.rename(os.path.join(character_folder, "1.png"), os.path.join(character_folder, "1357.png"))
    print(', ok.')


#--- CONCLUSION ---#
print(f'elapsed time: {(time.time() - t):.3f} s')