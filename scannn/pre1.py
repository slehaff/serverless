import PIL
from PIL import Image

import os
import numpy as np
import cv2

path = './'
# files = os.listdir(path+'slicedin')
# print(files)
# i = 37
# for file in files:
#     os.rename('slicedin/' + file, path + 'slicedin/' + str(i) + '.png')
#     os.rename('slicedout/' + file, 'slicedout/'+str(i)+'.png')
#     i += 1
#     print(file)
n = 38
for i in range(8):
    for j in range(7):
        count = i*10+j
        os.rename('slicedin/' + str("%02d" % count) + '.png',
                  path + 'slin/' + str(n) + '.png')
        os.rename('slicedout/' + str("%02d" % count) + '.png',
                  path + 'slout/'+str(n)+'.png')
        n += 1
        print(n)
