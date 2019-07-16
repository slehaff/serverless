import PIL
from PIL import Image

import os
import numpy as np
import cv2


def rename():
    path = './'
    files = os.listdir(path+'inp/')
    i = 0
    for f in files:
        # img = Image.open(path+f)
        # img = img.resize((74, 56))
        # img.save(path+f)
        os.rename('inp/' + f, path + 'inp/' + str(i) + '.png')
        os.rename('out/' + f, 'out/'+str(i)+'.png')
        i += 1
        print(f)


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = np.array(img)
    img = img.astype(np.float32)
    img = img - np.min(img)
    img = img/np.max(img)
    img = img*255
    img = Image.fromarray(img.astype('uint8'))
    return img


def cropme():
    path = './'
    files = os.listdir(path+'inp/')
    i = 0
    for f in files:
        slice = Image.open(path + 'inp/'+f)
        slice = normalize_image(slice)
        slice = slice.crop((0, 0, 64, 64))
        slice.save(path + 'inp2/'+f)
        slice = Image.open(path + 'out/'+f)
        slice = normalize_image(slice)
        slice = slice.crop((0, 0, 64, 64))
        slice.save(path + 'out2/'+f)


def rotateall():
    path = './'
    files = os.listdir(path+'inpplus/')
    i = 0
    for f in files:
        slice = Image.open(path + 'inpplus/'+f)
        slice = slice.rotate(90)
        fn = int(f[:-4])+163
        fout = str(fn) + '.png'
        slice.save(path + 'inpplus/'+fout)
        slice = Image.open(path + 'outplus/'+f)
        slice = slice.rotate(90)
        fn = int(f[:-4])+163
        fout = str(fn) + '.png'
        slice.save(path + 'outplus/'+fout)
        print(fout)


rotateall()
