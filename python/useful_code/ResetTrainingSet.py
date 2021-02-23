import os

TRAINING_SET_OUTPUT_PATH = r'C:\Users\franc\Desktop\training_set'

for i in range(10):
    dir = os.path.join(TRAINING_SET_OUTPUT_PATH, str(i))
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
