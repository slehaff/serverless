import numpy as np
import os
import cv2

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras import optimizers

import matplotlib.pyplot as plt


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def filter_image_sobelx(img):
    # Perform filtering to the input image
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    return sobelx


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
data_dir = './tfdir/data/'
folderpaths = [os.path.join(data_dir, o) for o in os.listdir(
    data_dir) if os.path.isdir(os.path.join(data_dir, o))]
imagepaths = []

for folderpath in folderpaths:
    temppaths = [os.path.join(folderpath, fname) for fname in os.listdir(
        folderpath) if fname.endswith('.jpg')]
    imagepaths += temppaths

# Load and pre-process the training data
images = []
grayimages = []
filteredimages = []

np.random.shuffle(imagepaths)
for imagepath in imagepaths:
    print(imagepath)
    img = cv2.imread(imagepath).astype(np.float32)
    img = normalize_image255(img)
    gray_img = make_grayscale(img)
    filtered_img = filter_image_sobelx(gray_img)
    images.append(img)
    grayimages.append(gray_img)
    filteredimages.append(filtered_img)

images = np.array(images, dtype='float32')
grayimages = np.array(grayimages, dtype='float32')
filteredimages = np.array(filteredimages, dtype='float32')

# Expand the image dimension to conform with the shape required by keras and tensorflow, inputshape=(..., h, w, nchannels).
grayimages = np.expand_dims(grayimages, -1)
filteredimages = np.expand_dims(filteredimages, -1)

print("images shape: {}".format(images.shape))
print("grayimages shape: {}".format(grayimages.shape))
print("filteredimages shape: {}".format(filteredimages.shape))
print(len(images))

# Visualize an arbitrary image and the filtered version of it
margin_img = np.ones(shape=(256, 10, 3))
combined_image = np.hstack((img, margin_img, np.dstack(
    (gray_img,)*3), margin_img, np.dstack((normalize_image(filtered_img),)*3)))

cv2.imwrite('OriginalGrayFiltered_sobelx.png',
            (255.0*combined_image).astype(np.uint8))


input_height, input_width = gray_img.shape


def linearcnn_model():
    # Returns a convolutional neural network model with a single linear convolution layer
    model = Sequential()
    model.add(Conv2D(1, (3, 3), padding='same',
                     input_shape=(input_height, input_width, 1)))
    return model


model = linearcnn_model()
sgd = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
model.summary()

number_of_epochs = 5
loss = []
val_loss = []
convweights = []


for epoch in range(number_of_epochs):
    history_temp = model.fit(grayimages, filteredimages,
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

predicted_img = model.predict(np.array([np.expand_dims(gray_img, -1)]))
predicted_img = predicted_img.squeeze()

margin_img = np.ones(shape=(256, 10, 3))
combined_image = np.hstack((np.dstack((normalize_image(
    predicted_img),)*3), margin_img, np.dstack((normalize_image(filtered_img),)*3)))

cv2.imwrite('PredictedFiltered_sobelx.png',
            (255.0*combined_image).astype(np.uint8))
