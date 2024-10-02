#!/usr/bin/env python
# -*- coding: utf-8 -*-
from symbol import import_from

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np



def build_ROSS_32_50(cfg):

    """
    input_shape -> (height, width, channel)
    """
    img_input = keras.Input(shape=cfg.input_shape)

    # Block 1:
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(img_input)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    #print('Block 1:', x.shape)

    # Block 2:
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    #print('Block 2:', x.shape)

    # Block 3:
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    #print('Block 3:', x.shape)

    # Block 4:
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    if cfg.GT_Output_shape == (32, 2):
        x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    elif cfg.GT_Output_shape == (16, 2) or cfg.GT_Output_shape == (8, 2):
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    #print('Block 4:', x.shape)

    x = layers.Dropout(cfg.HP_DROPOUT)(x)

    # Block 5:
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    if cfg.GT_Output_shape == (8, 2):
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    else:
        x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    #print('Block 5:', x.shape)
    x = layers.Dropout(cfg.HP_DROPOUT)(x)

    # Block 6:
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = keras.layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    #print('Block 6:', x.shape)
    x = layers.Dropout(cfg.HP_DROPOUT)(x)

    # Block 7:
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    #print('Block 7:', x.shape)

    # Block 8
    if cfg.Radar_Range <= 25:
        x = layers.Conv2D(2048, (1, 1), strides=(1, 1), padding="valid")(x)
    else:
        x = layers.Conv2D(2048, (2, 1), strides=(1, 1), padding="valid")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(2048, (1, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(2048, (1, 1), strides=(1, 1))(x)
    x = layers.ELU()(x)
    #print('Block 8:', x.shape)

    x = layers.Dropout(cfg.HP_DROPOUT)(x)
    #print(x.shape)

    x = layers.Conv2D(cfg.Output_vertices, (1, 1), strides=(1, 1), activation="softmax")(x)
    #print(x.shape)
    x = layers.Reshape((cfg.GT_Output_shape[0], cfg.Output_vertices))(x)
    #print(x.shape)

    model = tf.keras.models.Model(inputs=img_input, outputs=x)

    return model

#import ROSS.cfg.ROSS_Config as cfg

#print(cfg.input_shape)

#build_ROSS_32_50(cfg)