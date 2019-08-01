# Cnn2 as per paper, processing to generate cos and sin Nom and denom
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras import layers
from keras.models import Sequential, Model, Input
from keras import optimizers
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Lambda
from keras.layers import Conv2D, MaxPooling2D, Add, UpSampling2D, concatenate, Concatenate, AveragePooling2D
from keras.utils import plot_model

from nnwrap import *

number_of_epochs = 100
IMAGECOUNT = 150


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)number_of_epochs = 5
    img = img/255.0
    return img


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img


def to_array(folder_path, array, file_count):
    for i in range(0, file_count, 1):
        myfile = folder_path + str(i)+'.png'
        img = cv2.imread(myfile).astype(np.float32)
        img = normalize_image255(img)
        inp_img = make_grayscale(img)
        array.append(inp_img)
    return
   


def to_npy_array(folder_path, array, file_count):
    for i in range(0, file_count, 1):
        myfile = folder_path + str(i)+'.npy'
        img = np.load(myfile)
        img = normalize_image255(img)
        array.append(img)
        print('npyarray shape:', np.shape(array))
    return


# Load and pre-process the training data
fringe_images = []
background_images = []
nom_images = []
denom_images = []

to_array('newfringeA/', fringe_images, IMAGECOUNT)
to_array('newgray/', background_images, IMAGECOUNT)
to_npy_array('newnom/', nom_images, IMAGECOUNT)
to_npy_array('newdenom/', denom_images, IMAGECOUNT)


input_height = 170
input_width = 170


# Expand the image dimension to conform with the shape required by keras and tensorflow, inputshape=(..., h, w, nchannels).
fringe_images = np.expand_dims(fringe_images, -1)
background_images = np.expand_dims(background_images, -1)
# nom_images = np.expand_dims(nom_images, -1)
# denom_images = np.expand_dims(denom_images, -1)
print('fringeshape:', np.shape(fringe_images ))
print('backgroundshape:', np.shape(background_images) )
print('nomshape:', np.shape(nom_images ))
print('denomshape:', np.shape(denom_images) )

print("input shape: {}".format(fringe_images.shape))
# print("output shape: {}".format(nom_images.shape))
print(len(fringe_images))

fringe_image_A = Input(shape=(input_height, input_width, 1))
background_image_B = Input(shape=(input_height, input_width, 1))

# ============================= Branch 1 Fringe Image ================================

A1 = Conv2D(50, (3, 3), padding='same')(fringe_image_A)
# A2 = BatchNormalization(axis=-1)(A1)
A3 = Activation('relu')(A1)

# Begin 4 Res Block A ====================================================================

A4 = Conv2D(50, (3, 3), padding='same')(A3)
# A5 = BatchNormalization(axis=-1)(A4)
A6 = Activation('relu')(A4)
A7 = Conv2D(50, (3, 3), padding='same')(A6)
# A8 = BatchNormalization(axis=-1)(A7)
A9 = layers.add([A3, A7])
A10 = Activation('relu')(A9)

A11 = Conv2D(50, (3, 3), padding='same')(A10)    # predicted_img[0] = predicted_img[0].squeeze()
    # predicted_img[1] = predicted_img[1].squeeze()
# A12 = BatchNormalization(axis=-1)(A11)
A13 = Activation('relu')(A11)
A14 = Conv2D(50, (3, 3), padding='same')(A13)    # predicted_img[0] = predicted_img[0].squeeze()
    # predicted_img[1] = predicted_img[1].squeeze()
# A15 = BatchNormalization(axis=-1)(A14)
A16 = layers.add([A10, A14])
A17 = Activation('relu')(A16)

A18 = Conv2D(50, (3, 3), padding='same')(A17)    # predicted_img[0] = predicted_img[0].squeeze()
    # predicted_img[1] = predicted_img[1].squeeze()
# A19 = BatchNormalization(axis=-1)(A18)
A20 = Activation('relu')(A18)
A21 = Conv2D(50, (3, 3), padding='same')(A20)    # predicted_img[0] = predicted_img[0].squeeze()
    # predicted_img[1] = predicted_img[1].squeeze()
# A22 = BatchNormalization(axis=-1)(A21)
A23 = layers.add([A17, A21])
A24 = Activation('relu')(A23)

A25 = Conv2D(50, (3, 3), padding='same')(A24)
# A26 = BatchNormalization(axis=-1)(A25)
A27 = Activation('relu')(A25)
A28 = Conv2D(50, (3, 3), padding='same')(A27)
# A29 = BatchNormalization(axis=-1)(A28)
A30 = layers.add([A24, A28])
A31 = Activation('relu')(A30)

# End 4 Res Block ====================================================================

A32 = Conv2D(50, (3, 3), padding='same')(A31)
# A33 = BatchNormalization(axis=-1)(A32)
output_A = Activation('relu')(A32)


# ============================== End of Branch A =======================================

# =======================Branch 2 No Fringe Background=================================

B1 = Conv2D(50, (3, 3), padding='same')(background_image_B)
# B2 = BatchNormalization(axis=-1)(B1)
B3 = Activation('relu')(B1)

pooldown = AveragePooling2D(pool_size=(2, 2))(B3)

# Begin 4 Res Block B ====================================================================

B4 = Conv2D(50, (3, 3), padding='same')(pooldown)
# B5 = BatchNormalization(axis=-1)(B4)
B6 = Activation('relu')(B4)
B7 = Conv2D(50, (3, 3), padding='same')(B6)
# B8 = BatchNormalization(axis=-1)(B7)
B9 = layers.add([pooldown, B7])
B10 = Activation('relu')(B9)

B11 = Conv2D(50, (3, 3), padding='same')(B10) 
# B12 = BatchNormalization(axis=-1)(B11)
B13 = Activation('relu')(B11)
B14 = Conv2D(50, (3, 3), padding='same')(B13) 
# B15 = BatchNormalization(axis=-1)(B14)
B16 = layers.add([B10, B14])
B17 = Activation('relu')(B16)

B18 = Conv2D(50, (3, 3), padding='same')(B17)
# B19 = BatchNormalization(axis=-1)(B18)
B20 = Activation('relu')(B18)
B21 = Conv2D(50, (3, 3), padding='same')(B20)
# B22 = BatchNormalization(axis=-1)(B21)
B23 = layers.add([B17, B21])
B24 = Activation('relu')(B23)

B25 = Conv2D(50, (3, 3), padding='same')(B24)
# B26 = BatchNormalization(axis=-1)(B25)
B27 = Activation('relu')(B25)
B28 = Conv2D(50, (3, 3), padding='same')(B27)
# B29 = BatchNormalization(axis=-1)(B28)
B30 = layers.add([B24, B28])
B31 = Activation('relu')(B30)

# End 4 Res Block ====================================================================

poolup = UpSampling2D(size=(2, 2), interpolation='nearest')(B31)
B32 = Activation('relu')(poolup)
B33 = Conv2D(50, (3, 3), padding='same')(B32)
# B34 = BatchNormalization(axis=-1)(B33)
output_B = Activation('relu')(B33)

# ============================== End of Branch B =======================================

# ================================= Output Merge A a./testtf/data/==============================

combined = concatenate([output_A, output_B], axis=3)
x = Conv2D(2, (3, 3), padding='same')(combined)
# x1 = Flatten()(x)
y_nom = Lambda(lambda x: x[:, :, :, 0])(x)
y_denom = Lambda(lambda x: x[:, :, :, 1])(x)

# ================================= End of Merge ======================================

cnn2_model = Model(inputs=[fringe_image_A, background_image_B], outputs=[
                   y_nom, y_denom])
cnn2_model.summary()


def compile_model(model):
    # model = Model(input_image, output_image)
    sgd = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()
    return(model)


loss = []
val_loss = []
convweights = []

compile_model(cnn2_model)
model = cnn2_model
plot_model(cnn2_model, show_shapes=True, to_file='models/cnn2_model.png')


def load_model():
    model = keras.models.load_model('models/cnn2a-bmodel-shd-npy-150-20.h5')
    return(model)


# model = load_model()


def fct_train():
    for epoch in range(number_of_epochs):
        
        print('epoch #:', epoch)
        history_temp = model.fit([fringe_images, background_images], [nom_images, denom_images],
                                 batch_size=4,
                                 epochs=1,
                                 validation_split=0.2)
        loss.append(history_temp.history['loss'][0])
        val_loss.append(history_temp.history['val_loss'][0])
        # convweights.append(model.layers[0].get_weights()[0].squeeze())


fct_train()


def plot():
    # Plot the training and validation losses
    plt.close('all')

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.show()
    plt.savefig('trainingvalidationlossgx.png')


plot()


def combImages(x1, x2, i1, i2, i3, i4):
    new_img = img4 = np.concatenate((x1, x2, i1, i2, i3, i4), axis=1)
    return(new_img)


def DB_predict(i, x1, x2, y1, y2):
    print('y1 shape:', y1.shape)
    predicted_img = model.predict(
        [np.array([np.expand_dims(x1, -1)]), np.array([np.expand_dims(x2, -1)])])
    predicted_img[0] = predicted_img[0].squeeze()
    predicted_img[1] = predicted_img[1].squeeze()
    # wrap = nn_wrap(predicted_img[0], predicted_img[1]) # use prediction output
    saveswat(i, predicted_img[0], predicted_img[1])
    # nnnom, nndenom = loadswat(i)

    # wrap = nn_wrap(255.0*y1, 255.0*y2) # Use scanning output   
    # cv2.imwrite('validate/'+str(i)+'filteredSync.png',
    #             (255.0*predicted_img[0]).astype(np.uint8))
    # cv2.imwrite('validate/'+str(i)+'input.png',
    #             (255.0*predicted_img[1]).astype(np.uint8))
    combo = combImages(255.0*x1, 255.0*x2, 255.0* y1, 255.0 *
                       predicted_img[0], 255.0*y2, 255.0*predicted_img[1])

    # cv2.imwrite('validate/'+str(i)+'combo.png', (1.0*combo).astype(np.uint8))
    return(combo)


# get_my_file('inp/' + str(1)+'.png')
myfile = 'fringeA/' + str(1)+'.png'

img = cv2.imread(myfile).astype(np.float32)
img = normalize_image255(img)
inp_img = make_grayscale(img)
combotot = combImages(inp_img, inp_img, inp_img, inp_img, inp_img, inp_img)
for i in range(0, 150, 1):
    print(i)
    # get_my_file('inp/' + str(i)+'.png')
    myfile = 'newfringeA/' + str(i)+'.png'
    img = cv2.imread(myfile).astype(np.float32)
    inp_1 = normalize_image255(img)
    inp_1 = make_grayscale(inp_1)

    myfile = 'newgray/' + str(i)+'.png'
    img = cv2.imread(myfile).astype(np.float32)
    inp_2 = normalize_image255(img)
    inp_2 = make_grayscale(inp_2)

    myfile = 'newnom/' + str(i)+'.npy'
    nom_img = np.load(myfile)
    # nom_img = normalize_image255(nom_img)

    myfile = 'newdenom/' + str(i)+'.npy'
    denom_img = np.load(myfile)
    # denom_img = normalize_image255(denom_img)

    combo = DB_predict(i, inp_1, inp_2, nom_img, denom_img)
    combotot = np.concatenate((combotot, combo), axis=0)

save1nnwrap()
save1wrap()
model.save('models/cnn2a-bmodel-shd-npy-150-100.h5')
cv2.imwrite('validate/'+'cnn2a-shd-npy-150-100-0.png',
            (1.0*combotot).astype(np.uint8))
