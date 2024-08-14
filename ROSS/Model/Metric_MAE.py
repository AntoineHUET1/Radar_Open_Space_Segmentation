import tensorflow as tf
class Metric_MAE(tf.keras.metrics.Metric):
    def __init__(self, cfg, name='custom_mae_metric', **kwargs):
        super(Metric_MAE, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='mae', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.Radar_Range = cfg.Radar_Range
        self.GT_mode = cfg.GT_mode
        self.mae_fn = tf.keras.losses.MeanAbsoluteError()  # Instantiate the MAE loss function

    def update_state(self, target, predict, sample_weight=None):
        have_target, pos = tf.split(target, 2, axis=-1)
        mask = tf.cast(have_target, dtype=tf.bool)

        predict = tf.argmax(predict, axis=-1)
        predict = tf.cast(predict, dtype=tf.float32)
        predict = tf.expand_dims(predict, axis=-1)

        if self.GT_mode == 0: # Full GT
            mae = self.mae_fn(pos, predict)
        else: # Only obstacles in range
            masked_true_values = tf.boolean_mask(pos, mask)
            masked_pred_values = tf.boolean_mask(predict, mask)
            mae = self.mae_fn(masked_true_values, masked_pred_values)  # Compute MAE using the instance

        # Normalize the MAE by the Range
        mae = mae*self.Radar_Range/50

        self.mae.assign_add(mae)
        self.count.assign_add(1)

    def result(self):
        return self.mae / self.count

    def reset_state(self):
        self.mae.assign(0)
        self.count.assign(0)