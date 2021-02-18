import os
from cv2 import cv2
import numpy

TRAINING_SET_PATH = r'C:\Users\franc\Desktop\training_set'
OUTPUT_PATH = r'C:\Users\franc\Desktop\minmax_set'

def getClosestDigitMinMax(image, min_digits, max_digits):
    has_min = [(min <= image).all() for min in min_digits]
    is_in_max = [(image <= max).all() for max in max_digits]
    
    valid = -1
    for i in range(10):
        if(has_min[i] and is_in_max[i]):
            if(valid == -1):
                valid = i
            else:
                return -1
    return valid

if(os.path.isdir(OUTPUT_PATH) == False):
    os.mkdir(OUTPUT_PATH)

first_image = cv2.imread(os.path.join(OUTPUT_PATH, str(0), "min.png"))
min_image = numpy.zeros((first_image.shape[0], first_image.shape[1]*10, first_image.shape[2]), dtype=first_image.dtype)
max_image = numpy.zeros((first_image.shape[0], first_image.shape[1]*10, first_image.shape[2]), dtype=first_image.dtype)

min_images = [cv2.cvtColor(cv2.imread(os.path.join(OUTPUT_PATH, str(i), "min.png")), cv2.COLOR_BGR2GRAY) for i in range(10)]
max_images = [cv2.cvtColor(cv2.imread(os.path.join(OUTPUT_PATH, str(i), "max.png")), cv2.COLOR_BGR2GRAY) for i in range(10)]

for i in range(10):
    if((min_images[i][numpy.where(min_images[i] != 0)] != 255).any()):
        print(f'{i} has not polarized values')
    for j in range(10):
        if(i == j and (min_image[i] > max_image[j]).any() == True):
            print(f'{i} max isn\'t a subset of min')
        elif(i != j and (min_images[i] > max_images[j]).any() == False):
            print(f'{i} min is a subset of {j} max')

    # min_image[:, i*first_image.shape[1] : (i+1)*first_image.shape[1]] = cv2.imread(os.path.join(OUTPUT_PATH, str(i), "min.png"))
    # max_image[:, i*first_image.shape[1] : (i+1)*first_image.shape[1]] = cv2.imread(os.path.join(OUTPUT_PATH, str(i), "max.png"))


    # print(i)
    # dir = os.path.join(TRAINING_SET_PATH, str(i))
    # first_file = os.listdir(dir)[0]
    # min_image = cv2.cvtColor(cv2.imread(os.path.join(dir, first_file)), cv2.COLOR_BGR2GRAY)
    # max_image = min_image.copy()
    # for f in os.listdir(dir):
    #     curr_image = cv2.imread(os.path.join(dir, f))
    #     curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    #     min_image = numpy.minimum(min_image, curr_image)
    #     max_image = numpy.maximum(max_image, curr_image)
    
    # if(os.path.isdir(os.path.join(OUTPUT_PATH, str(i))) == False):
    #     os.mkdir(os.path.join(OUTPUT_PATH, str(i)))
    # cv2.imwrite(os.path.join(OUTPUT_PATH, str(i), "min.png"), min_image)
    # cv2.imwrite(os.path.join(OUTPUT_PATH, str(i), "max.png"), max_image)
print('puff')
# cv2.imwrite(os.path.join(OUTPUT_PATH, "min.png"), min_image)
# cv2.imwrite(os.path.join(OUTPUT_PATH, "max.png"), max_image)