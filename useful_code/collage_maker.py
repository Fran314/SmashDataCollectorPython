import os, sys
import cv2
import numpy
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import functions as fun

character_infos = fun.readTSV(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\characters_info.tsv')

temp_image = cv2.imread(os.path.join(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\23-coded_faces', character_infos[0][1], "0.png"))
tile_width = temp_image.shape[1]
tile_height = temp_image.shape[0]
closeups = numpy.zeros((tile_height * len(character_infos), tile_width * 8, 4))
for i in range(len(character_infos)):
    character = character_infos[i]
    for j in range(8):
        temp_image = cv2.cvtColor(cv2.imread(os.path.join(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\23-coded_faces', character[1], str(j) + ".png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
        closeups[i * tile_height : (i+1) * tile_height, j * tile_width : (j+1) * tile_width] = temp_image
cv2.imwrite(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\23-closeups.png', closeups)

temp_image = cv2.imread(os.path.join(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\4-coded_faces', character_infos[0][1], "0.png"))
tile_width = temp_image.shape[1]
tile_height = temp_image.shape[0]
closeups = numpy.zeros((tile_height * len(character_infos), tile_width * 8, 4))
for i in range(len(character_infos)):
    character = character_infos[i]
    for j in range(8):
        temp_image = cv2.cvtColor(cv2.imread(os.path.join(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\4-coded_faces', character[1], str(j) + ".png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
        closeups[i * tile_height : (i+1) * tile_height, j * tile_width : (j+1) * tile_width] = temp_image
cv2.imwrite(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\4-closeups.png', closeups)

temp_image = cv2.imread(os.path.join(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\icons', character_infos[0][1], "1.png"))
tile_width = temp_image.shape[1]
tile_height = temp_image.shape[0]
icons = numpy.zeros((tile_height * len(character_infos), tile_width * 8, 4))
for i in range(len(character_infos)):
    character = character_infos[i]
    for j in range(8):
        temp_image = cv2.cvtColor(cv2.imread(os.path.join(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\temp\icons', character[1], str(j+1) + ".png"), flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
        icons[i * tile_height : (i+1) * tile_height, j * tile_width : (j+1) * tile_width] = temp_image
cv2.imwrite(r'C:\Users\franc\Documents\VSCode\SmashDataCollector\res\character_references\icons.png', icons)