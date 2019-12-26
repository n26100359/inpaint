from __future__ import print_function
import sklearn
import keras
import math
import random
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Reshape, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, UpSampling2D, BatchNormalization, concatenate, Lambda
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd

epochs = 300
batch_size = 10

# input image dimensions
img_rows, img_cols = 32, 32

# the data
xtrain32 = np.load('./xtrain32.npy',allow_pickle=True) / 255
ytrain32 = np.load('./ytrain32.npy',allow_pickle=True) / 255
xtest32 = np.load('./xtest32.npy',allow_pickle=True) / 255
ytest32 = np.load('./ytest32.npy',allow_pickle=True) / 255

xtrain96 = np.load('./xtrain96.npy',allow_pickle=True) / 255
ytrain96 = np.load('./ytrain96.npy',allow_pickle=True) / 255
xtest96 = np.load('./xtest96.npy',allow_pickle=True) / 255
ytest96 = np.load('./ytest96.npy',allow_pickle=True) / 255

x_train = np.concatenate((xtrain32,ytrain32),axis=2)
x_test = np.concatenate((xtest32,ytest32),axis=2)
y_train = np.concatenate((xtrain96,ytrain96),axis=2)
y_test = np.concatenate((xtest96,ytest96),axis=2)

print(x_train.shape)
input_shape = (32, 32, 3)
inputs = Input(input_shape)
conv1 = Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(inputs)
conv2 = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(conv1)
conv3 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(conv2)
f = Flatten()(conv3)
d1 = Dense(128, activation='relu')(f)
d2 = Dense(256, activation='relu')(d1)
d3 = Dense(512, activation='relu')(d2)
d4 = Dense(32*32*30, activation='relu')(d3)
rs = Reshape((32,32,30))(d4)
up = UpSampling2D((3,3))(rs)
tpconv1 = Conv2DTranspose(64, kernel_size=(3,3), padding='same', activation='relu')(up)
tpconv2 = Conv2DTranspose(32, kernel_size=(3,3), padding='same', activation='relu')(tpconv1)
tpconv3 = Conv2DTranspose(16, kernel_size=(3,3), padding='same', activation='relu')(tpconv2)
out = Conv2DTranspose(3, kernel_size=(3,3), padding='same', activation='relu')(tpconv3)

                                        
                                       
model = Model(inputs, out, name='Enlarger')
model.summary()
model.compile(loss='mse',
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          )
model.save('./models/enlarger_model/test.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])