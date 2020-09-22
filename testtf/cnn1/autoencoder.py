# Autoencoder module for instant depth determination

from keras.layers import Input, Dense, MaxPooling2D, Conv2D, Activation, UpSampling2D, add

A1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')

A = Input(shape=(H,W,1))
A1 = Conv2D(32,(3,3), padding='same')(A)
A2 = Activation('relu')(A1)
A3 = Conv2D(32,(3,3), padding='same')(A2)
A4 = Activation('relu')(A3)

A5 = MaxPooling2D(pool_size=(2,2), stride=(1,1), padding='same')(A4)
A6 = Conv2D(64,(3,3), padding='same')(A5)
A7 = Activation('relu')(A6)
A8 = Conv2D(64,(3,3), padding='same')(A7)
A9 = Activation('relu')(A8)

A10 = MaxPooling2D(pool_size=(2,2), stride=(1,1), padding='same')(A9)
A11 = Conv2D(128,(3,3), padding='same')(A10)
A12 = Activation('relu')(A11)
A13 = Conv2D(128,(3,3), padding='same')(A12)
A14 = Activation('relu')(A13)

A10 = MaxPooling2D(pool_size=(2,2), stride=(1,1), padding='same')(A9)
A11 = Conv2D(256,(3,3), padding='same')(A10)
A12 = Activation('relu')(A11)
A13 = Conv2D(256,(3,3), padding='same')(A12)
A14 = Activation('relu')(A13)

A10 = MaxPooling2D(pool_size=(2,2), stride=(1,1), padding='same')(A9)
A11 = Conv2D(512,(3,3), padding='same')(A10)
A12 = Activation('relu')(A11)
A13 = Conv2D(512,(3,3), padding='same')(A12)
A14 = Activation('relu')(A13)

A10 = MaxPooling2D(pool_size=(2,2), stride=(1,1), padding='same')(A9)
A11 = Conv2D(1024,(3,3), padding='same')(A10)
A12 = Activation('relu')(A11)
A13 = Conv2D(1024,(3,3), padding='same')(A12)
A14 = Activation('relu')(A13)

############################################# Decode ###############################################

A15 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A14)
A16 = add([A15, A13])
