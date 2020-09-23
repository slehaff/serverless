# Autoencoder module for instant depth determination

from keras.layers import Input, MaxPooling2D, Conv2D, Activation, UpSampling2D, add

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

A15 = MaxPooling2D(pool_size=(2,2), stride=(1,1), padding='same')(A14)
A16 = Conv2D(256,(3,3), padding='same')(A15)
A17 = Activation('relu')(A16)
A18 = Conv2D(256,(3,3), padding='same')(A17)
A19 = Activation('relu')(A18)

A20 = MaxPooling2D(pool_size=(2,2), stride=(1,1), padding='same')(A19)
A21 = Conv2D(512,(3,3), padding='same')(A20)
A22 = Activation('relu')(A21)
A23 = Conv2D(512,(3,3), padding='same')(A22)
A24 = Activation('relu')(A23)

A25 = MaxPooling2D(pool_size=(2,2), stride=(1,1), padding='same')(A24)
A26 = Conv2D(1024,(3,3), padding='same')(A25)
A27 = Activation('relu')(A26)
A28 = Conv2D(1024,(3,3), padding='same')(A27)
A29 = Activation('relu')(A28)

############################################# Decode ###############################################

A30 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A29)
A31 = add([A24, A30])
A32 = Conv2D(512,(3,3), padding='same')(A31)
A33 = Activation('relu')(A32)
A34 = Conv2D(512,(3,3), padding='same')(A33)
A35 = Activation('relu')(A34)

A36 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A35)
A37 = add([A24, A30])
A38 = Conv2D(256,(3,3), padding='same')(A37)
A39 = Activation('relu')(A38)
A40 = Conv2D(256,(3,3), padding='same')(A39)
A41 = Activation('relu')(A40)

A42 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A41)
A43 = add([A24, A30])
A44 = Conv2D(128,(3,3), padding='same')(A43)
A45 = Activation('relu')(A44)
A46 = Conv2D(128,(3,3), padding='same')(A45)
A47 = Activation('relu')(A46)

A48 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A47)
A49 = add([A24, A30])
A50 = Conv2D(64,(3,3), padding='same')(A49)
A51 = Activation('relu')(A50)
A52 = Conv2D(64,(3,3), padding='same')(A51)
A53 = Activation('relu')(A52)

A54 = UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(A53)
A55 = add([A24, A30])
A56 = Conv2D(32,(3,3), padding='same')(A55)
A57 = Activation('relu')(A56)
A58 = Conv2D(32,(3,3), padding='same')(A57)
A59 = Activation('relu')(A58)
# Output Conv2D(1)

