#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np



def build_ROS_32_50(input_shape=(256, 256, 1),Half_length=False,Mode=0,Dropout=0.2):

    """
    input_shape -> (height, width, channel)
    """
    img_input = keras.Input(shape=input_shape)

    # Block 1:
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(img_input)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    print('Conv 1',x.shape)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2:
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    print('Conv 2', x.shape)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3:
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    print('Conv 3', x.shape)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4:
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    print('Conv 4', x.shape)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    print('Conv 4.1', x.shape)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)


    x = layers.Dropout(Dropout)(x)

    # Block 5:
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    print('Conv 5', x.shape)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(Dropout)(x)

    # Block 6:
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = keras.layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    print('Conv 6', x.shape)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(Dropout)(x)

    # Block 7:
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    print('Conv 7', x.shape)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    # Block 8
    print('Conv 8', x.shape)
    if Half_length:
        x = layers.Conv2D(2048, (1, 1), strides=(1, 1), padding="valid")(x)
    else:
        x = layers.Conv2D(2048, (2, 1), strides=(1, 1), padding="valid")(x)
    x = layers.ELU()(x)
    print('Conv 8.1', x.shape)
    x = layers.Conv2D(2048, (1, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    print('Conv 8.2', x.shape)
    x = layers.Conv2D(2048, (1, 1), strides=(1, 1))(x)
    x = layers.ELU()(x)
    print('Conv 8.3', x.shape)

    x = layers.Dropout(Dropout)(x)


    if Mode==0:
        # Replace the classification layers with regression layers
        x = layers.Conv2D(1, (1, 1), strides=(1, 1), activation="linear")(x)
        print('Conv 8.4', x.shape)
        x = layers.Reshape((32, 1))(x)
        print('Conv 8.5', x.shape)
    else:

        x = layers.Conv2D(50, (1, 1), strides=(1, 1), activation="softmax")(x)
        print('Conv 8.4', x.shape)
        x = layers.Reshape((32, 50))(x)
        print('Conv 8.5', x.shape)

    model = tf.keras.models.Model(inputs=img_input, outputs=x)

    return model

#model= build_ROS_32_50(input_shape=(128, 256, 1),Half_length=True,Mode=1)
#model.summary()
