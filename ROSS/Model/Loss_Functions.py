#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.losses import Loss
import keras.backend as K


class Custom_Loss(Loss):
    def __init__(
            self,
            #lAbel_size,
            #num_bins,
            alpha=1.0,
            epsilon=0.0001,
            entropy_weight=0.1
    ):
        super(Custom_Loss, self).__init__(name="stixel_loss")
        self._alpha = alpha
        self._epsilon = epsilon
        self._entropy_weight = entropy_weight

    def call(self, target, predict):
        have_target, stixel_pos = tf.split(target, 2, axis=-1)

        # Reshape y_true to match the shape of y_pred
        y_true = tf.reshape(stixel_pos, (-1,))

        # Calculate the cross-entropy loss for each prediction
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, predict, from_logits=False)

        # Average the loss over the batch and sequence dimensions
        loss = tf.reduce_mean(loss)

        return loss