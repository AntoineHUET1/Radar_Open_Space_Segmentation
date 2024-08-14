from tensorflow import keras
import tensorflow as tf
class CustomFit(keras.Model):
    def __init__(self, model,acc_metric):
        super(CustomFit, self).__init__()
        self.model = model
        self.best_val_loss = float('inf')
        self.best_val_acc = float('inf')

    def compile(self, optimizer, loss):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def build(self, input_shape):
        self.model.build(input_shape)
        super(CustomFit, self).build(input_shape)

    @tf.function
    def train_step(self, data):
        acc_metric.reset_state()
        x, y = data

        with tf.GradientTape() as tape:  # Forward Propagation
            y_pred = self.model(x, training=True)
            loss = self.loss(y, y_pred)

        trainable_vars = self.model.trainable_variables  # Get all trainable variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))  # Backward Propagation
        acc_metric.update_state(y, y_pred)
        return {"loss": loss, "accuracy": acc_metric.result()}

    @tf.function
    def test_step(self, data):
        acc_metric.reset_state()
        x, y = data
        y_pred = self.model(x, training=False)
        loss = self.loss(y, y_pred)
        acc_metric.update_state(y, y_pred)
        return {"loss": loss, "accuracy": acc_metric.result()}

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training, mask=mask)

    def save_if_best(self, val_loss, val_acc, filepath):

        if val_loss < self.best_val_loss or val_acc < self.best_val_acc:
            print(
                f"Validation loss improved => Saving model")  # "#from {self.best_val_loss:.3f} to {val_loss:.3f} => Saving model.")
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.model.save_weights(filepath)
        else:
            print(f"Validation loss did not improve")  # from {self.best_val_loss}.")