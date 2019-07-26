import numpy as np
# import pandas as pd
# import sklearn as sk
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import random
import math
import keras as k
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, Input, Dense, concatenate, Flatten, Add, Activation, BatchNormalization, Reshape
from keras import Model
import tensorflow as tf

# x = ...
# y = ...
# you shall load data and make sure that X has shape (samples, 12634) and y has shape (samples, 1)


# 1D residual block
def res1d(x1, layers, kernel=(3,), strides=1,):    
    assert K.ndim(x1) == 3

    ConvLayer = k.layers.Conv1D        
    normalizer = BatchNormalization

    c1 = x1        
    c1 = ConvLayer(layers, kernel_size=kernel, use_bias=0, strides=strides, padding='same')(c1)        
    c1 = normalizer(gamma_initializer='zeros')(c1)

    o1 = x1
    if strides > 1 or K.int_shape(x1)[-1] != layers:
        if strides > 1:
            o1 = k.layers.ZeroPadding1D((0, strides-1))(o1)
            o1 = k.layers.AveragePooling1D(strides)(o1)
        o1 = ConvLayer(layers, kernel_size=1, use_bias=0, padding='same')(o1)
        o1 = normalizer()(o1)

    v1 = Add()([c1, o1])
    v1 = Activation('relu')(v1)        
    return v1


# build the model
# the first branch is a 64 nodes fully connected layer
# the second branch is a series of residual convolutional blocks
# After Concatenate, gives output by sigmoid
def build():
    inp = Input(shape=(12634,)) # length of input
    
    v2 = Dense(64, activation='relu')(inp)
    
    v1 = k.layers.Reshape((-1, 1))(inp)
    v1 = res1d(v1, 4, 3)
    
    #for i in (8, 16, 32, 64, 128, 256, 256, 512, 512):
    for i in (8, 16, 32, 64, 128, 256, 512, 1024):
        v1 = res1d(v1, i, 3)
        v1 = res1d(v1, i, 3, strides=2) # strides=2, downsampling

    v1 = k.layers.Flatten()(v1)
    v1 = k.layers.Concatenate(axis=-1)([v1, v2])
    v1 = Dense(32, activation='relu')(v1)
    v1 = Dense(1, activation='sigmoid')(v1)

    model = k.models.Model(inputs=inp, outputs=v1)        
    return model

model = build()
# you may run 'summary' method to see detail structures
# model.summary()

# use adam with 0.0001 learn rate, measure by binary crossentropy loss
model.compile(optimizer=k.optimizers.Adam(lr=0.0001), metrics=['acc'], loss='binary_crossentropy')
model.fit(x, y, verbose=1, epochs=30, batch_size=32)
