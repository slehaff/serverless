from keras import layers
import numpy as np
import os
import cv2
import pandas as pd

import keras
from keras.models import Sequential, Model, Input
from keras.layers import Conv2D
from keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Add


def make_grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


IMAGECOUNT = 200


def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)
    img = img/255.0
    return img


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img


def combImages(i1, i2, i3):
    new_img = img3 = np.concatenate((i1, i2, i3), axis=1)
    return(new_img)


def DB_predict(i, x, y):
    predicted_img = model.predict(np.array([np.expand_dims(x, -1)]))
    predicted_img = predicted_img.squeeze()
    cv2.imwrite('predictout/'+str(i)+'.png',
                (255.0*predicted_img).astype(np.uint8))
    # cv2.imwrite('validate/'+str(i)+'input.png',
    # (255.0*x).astype(np.uint8))
    combo = combImages(255.0*x, 255.0*predicted_img, 255.0*y)
    #cv2.imwrite('validate/'+str(i)+'combo.png', (1.0*combo).astype(np.uint8))
    return(combo)


def load_model():
    model = keras.models.load_model('models/cnn0a-200-model0300+50-adam.h5')
    model.summary()
    return(model)


model = load_model()


# get_my_file('inp/' + str(1)+'.png')
myfile = 'fringeA/' + str(1)+'.png'
img = cv2.imread(myfile).astype(np.float32)
img = normalize_image255(img)
inp_img = make_grayscale(img)
combotot = combImages(inp_img, inp_img, inp_img)
print('start')
for i in range(0, IMAGECOUNT, 1):
    # print(i)
    # get_my_file('inp/' + str(i)+'.png')
    myfile = 'fringeA/' + str(i)+'.png'
    img = cv2.imread(myfile).astype(np.float32)
    img = normalize_image255(img)
    inp_img = make_grayscale(img)
    #get_my_file('out/' + str(i)+'.png')
    myfile = 'gray/' + str(i)+'.png'
    img = cv2.imread(myfile).astype(np.float32)
    img = normalize_image255(img)
    out_img = make_grayscale(img)
    combo = DB_predict(i, inp_img, out_img)
    combotot = np.concatenate((combotot, combo), axis=0)
cv2.imwrite('predictions/'+'predict-inp-cnn0a-300+50-Adam.png',
            (1.0*combotot).astype(np.uint8))
