import numpy as np
import os
import cv2

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)
    img = img/255.0
    return img


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img


# Get the paths to the training images
data_path = './testtf/data/'


# Load and pre-process the training data
input_filenames = []
output_filenames = []
#filteredimages = []


input_images = []
output_images = []


input_filenames = os.listdir(data_path + 'inputstrip/')
for fn in input_filenames:
    print(fn)
    img = cv2.imread(data_path + 'inputstrip/' + fn).astype(np.float32)
    img = normalize_image255(img)
    inp_img = make_grayscale(img)
    input_images.append(inp_img)
    fno = str(int(fn[:-4])+1) + '.png'
    print(fn, fno)
    img = cv2.imread(data_path + 'outputstrip/' + fno).astype(np.float32)
    img = normalize_image255(img)
    out_img = make_grayscale(img)
    output_images.append(out_img)
input_height, input_width = inp_img.shape
b = inp_img[25:265, 180:500]
print(b's shape=', b.shape)

# output_filenames = os.listdir(data_path + 'outputstrip/')
# print(output_filenames)
# for fn in output_filenames:
#     print(fn)
#     img = cv2.imread(data_path + 'outputstrip/' + fn).astype(np.float32)
#     img = normalize_image255(img)
#     out_img = make_grayscale(img)
#     output_images.append(out_img)
# input_height, input_width = inp_img.shape

# Expand the image dimension to conform with the shape required by keras and tensorflow, inputshape=(..., h, w, nchannels).
input_images = np.expand_dims(input_images, -1)
output_images = np.expand_dims(output_images, -1)


print("input shape: {}".format(input_images.shape))
print("output shape: {}".format(output_images.shape))
print(len(input_images))


def linearcnn_model():
    # Returns a convolutional neural network model with a single linear convolution layer
    model = Sequential()
    model.add(Conv2D(5, (5, 5), padding='same', activation='relu',
                     input_shape=(input_height, input_width, 1)))
    model.add(Conv2D(5, (5, 5), padding='same', activation='relu'))
    model.add(Conv2D(5, (5, 5), padding='same', activation='relu'))
    model.add(Conv2D(1, (5, 5), padding='same'))
    return model


model = linearcnn_model()
sgd = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
model.summary()

number_of_epochs = 4
loss = []
val_loss = []
convweights = []


for epoch in range(number_of_epochs):
    print('epoch #:', epoch)
    history_temp = model.fit(input_images, output_images,
                             batch_size=4,
                             epochs=1,
                             validation_split=0.2)
    loss.append(history_temp.history['loss'][0])
    val_loss.append(history_temp.history['val_loss'][0])
    convweights.append(model.layers[0].get_weights()[0].squeeze())

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


predicted_img = model.predict(np.array([np.expand_dims(inp_img, -1)]))
predicted_img = predicted_img.squeeze()
cv2.imwrite('filteredSync.png',
            (255.0*predicted_img).astype(np.uint8))
cv2.imwrite('input.png',
            (255.0*inp_img).astype(np.uint8))
