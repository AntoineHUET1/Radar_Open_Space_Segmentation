#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.losses import Loss
import tensorflow.keras.backend as K


import tensorflow as tf
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K

class Custom_Loss_2(Loss):
    def __init__(
            self,
            alpha=1.0,
            epsilon=0.01,
            entropy_weight=0.1
    ):
        super(Custom_Loss_2, self).__init__(name="stixel_loss")
        self._alpha = alpha
        self._epsilon = epsilon
        self._entropy_weight = entropy_weight

    def call(self, target, predict):

        have_target, stixel_pos = tf.split(target, 2, axis=-1)

        print("have_target:", have_target.shape)
        print("stixel_pos:", stixel_pos.shape)
        print("predict:", predict.shape)
        breakpoint()

        stixel_pos = stixel_pos - 0.5

        stixel_pos = (
                (stixel_pos - tf.math.floor(stixel_pos))
                + tf.math.floor(stixel_pos)
                + self._epsilon
        )

        floor_indices = tf.cast(tf.math.floor(stixel_pos), dtype="int32")
        ceil_indices = tf.cast(tf.math.ceil(stixel_pos), dtype="int32")

        # Debug print statements
        #tf.print("stixel_pos:", stixel_pos, summarize=-1)
        #tf.print("floor_indices:", floor_indices, summarize=-1)
        #tf.print("ceil_indices:", ceil_indices, summarize=-1)
        #tf.print("predict shape:", tf.shape(predict))

        fp = tf.gather(predict, floor_indices, batch_dims=-1)
        cp = tf.gather(predict, ceil_indices, batch_dims=-1)

        p = fp * (tf.math.ceil(stixel_pos) - stixel_pos) + cp * (
                stixel_pos - tf.math.floor(stixel_pos)
        )

        loss = -K.log(p) * have_target
        loss = K.sum(loss) / K.sum(have_target)

        return loss * self._alpha

import tensorflow as tf
from tensorflow.keras.losses import Loss



import tensorflow as tf
'''
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, name='custom_loss'):
        super().__init__(name=name)
        self.cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):

        have_target, y_true_R = tf.split(y_true, 2, axis=-1)

        # Compute the categorical crossentropy loss
        ce_loss = self.cross_entropy_loss(y_true_R, y_pred)

        y_true_R = tf.squeeze(y_true_R, axis=-1)  # Remove the last dimension
        
        # Compute the distance penalty
        y_pred_indices = tf.argmax(y_pred, axis=-1)  # Get the predicted class indices
        distance_penalty = tf.reduce_mean(tf.abs(tf.cast(y_pred_indices, tf.float32) - tf.cast(y_true_R, tf.float32)))
        
        # Combine the losses
        total_loss = ce_loss + distance_penalty
        
        return total_loss
'''
    
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, name='custom_loss'):
        super().__init__(name=name)
        self.cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # Split y_true into have_target and y_true_R
        have_target, y_true_R = tf.split(y_true, 2, axis=-1)

        # Squeeze the last dimension from y_true_R to match the shape for SparseCategoricalCrossentropy
        y_true_R = tf.squeeze(y_true_R, axis=-1)

        # Compute the categorical crossentropy loss without reduction
        ce_loss = self.cross_entropy_loss(y_true_R, y_pred)

        # Mask the loss using have_target
        masked_ce_loss = tf.reduce_sum(ce_loss * tf.squeeze(have_target, axis=-1)) / tf.reduce_sum(tf.squeeze(have_target, axis=-1))

        # Compute the distance penalty
        y_pred_indices = tf.argmax(y_pred, axis=-1)  # Get the predicted class indices
        distance_penalty = tf.abs(tf.cast(y_pred_indices, tf.float32) - tf.cast(y_true_R, tf.float32))

        # Mask the distance penalty using have_target
        masked_distance_penalty = tf.reduce_sum(distance_penalty * tf.squeeze(have_target, axis=-1)) / tf.reduce_sum(tf.squeeze(have_target, axis=-1))

        # Combine the losses
        total_loss = masked_ce_loss + masked_distance_penalty

        return total_loss
    
class CustomLoss_binnary_images(tf.keras.losses.Loss):
    def __init__(self, name='custom_loss'):
        super().__init__(name=name)
        self.binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # Assuming y_true is of shape (Batch, 32, 1)
        _, y_true_R = tf.split(y_true, num_or_size_splits=2, axis=-1)
        #print(y_pred.shape)
        #print(y_true_R.shape)
        #breakpoint()

        
        # Calculate the binary cross-entropy loss for each prediction
        loss = self.binary_cross_entropy(y_true_R, y_pred)
        
        # Mean the loss over the batch and the 32 predictions
        loss = tf.reduce_mean(loss)
        
        return loss
    