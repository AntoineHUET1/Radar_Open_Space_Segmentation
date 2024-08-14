from tensorflow import keras
import tensorflow as tf
import os
from tqdm import tqdm

def log_to_file(log_file_path, epoch, mean_train_loss, mean_train_accuracy, mean_val_loss, mean_val_accuracy):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Epoch {epoch + 1}: Train Loss: {mean_train_loss:.3f}, Train Accuracy: {mean_train_accuracy:.3f}, "
                       f"Validation Loss: {mean_val_loss:.3f}, Validation Accuracy: {mean_val_accuracy:.3f}\n")

class CustomFit(keras.Model):
    def __init__(self, model,acc_metric):
        super(CustomFit, self).__init__()
        self.model = model
        self.best_val_loss = float('inf')
        self.best_val_acc = float('inf')
        self.acc_metric = acc_metric

    def compile(self, optimizer, loss):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def build(self, input_shape):
        self.model.build(input_shape)
        super(CustomFit, self).build(input_shape)

    @tf.function
    def train_step(self, data):
        self.acc_metric.reset_state()
        x, y = data

        with tf.GradientTape() as tape:  # Forward Propagation
            y_pred = self.model(x, training=True)
            loss = self.loss(y, y_pred)

        trainable_vars = self.model.trainable_variables  # Get all trainable variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))  # Backward Propagation
        self.acc_metric.update_state(y, y_pred)
        return {"loss": loss, "accuracy": self.acc_metric.result()}

    @tf.function
    def test_step(self, data):
        self.acc_metric.reset_state()
        x, y = data
        y_pred = self.model(x, training=False)
        loss = self.loss(y, y_pred)
        self.acc_metric.update_state(y, y_pred)
        return {"loss": loss, "accuracy": self.acc_metric.result()}

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


def retrieve_training_info(log_file_path):
    import re

    # Initialize variables
    last_epoch = None
    best_val_loss = float('inf')
    best_val_acc = -float('inf')
    best_epoch = None

    patience = 0

    # Regex pattern to match log entries
    pattern = re.compile(
        r"Epoch (\d+): Train Loss: (\d+\.\d+), Train Accuracy: (\d+\.\d+), Validation Loss: (\d+\.\d+), Validation Accuracy: (\d+\.\d+)")

    with open(log_file_path, "r") as log_file:
        for line in log_file:
            match = pattern.match(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_accuracy = float(match.group(3))
                val_loss = float(match.group(4))
                val_accuracy = float(match.group(5))

                # Update last epoch
                last_epoch = epoch

                # Update patience:
                patience += 1
                # Update best validation metrics
                if val_loss < best_val_loss or val_accuracy < best_val_acc:
                    best_val_loss = val_loss
                    best_val_acc = val_accuracy
                    best_epoch = epoch
                    patience = 0
    last_epoch=last_epoch+1
    # Return results
    return last_epoch, best_val_loss, best_val_acc, best_epoch ,patience

def Train_model(cfg, model, acc_metric, loss, train_dataloader, train_dataloader_length, val_dataloader,val_dataloader_length,
                                test_dataloader, test_dataloader_length, run_dir):

    # Load logs if they exist:
    log_file_path = os.path.join(run_dir, "training_log.txt")

    best_val_loss = float('inf')
    best_val_acc = float('inf')
    no_improvement_count = 0
    last_epoch=0

    # Resume training from logs if they exist
    if os.path.exists(log_file_path):

        # Print logs
        print("\nResuming training from logs\n")
        with open(log_file_path, "r") as log_file:
            print(log_file.read())
        # Load the last model weights
        model.load_weights(run_dir+"/last_model_weights.weights.h5")
        # Retrieve training info
        last_epoch, best_val_loss, best_val_acc, best_epoch,no_improvement_count = retrieve_training_info(log_file_path)

        if no_improvement_count >= cfg.patience:
            print(f"Training already finished at epoch {last_epoch} with best validation loss of {best_val_loss:.3f} and best validation accuracy of {best_val_acc:.3f}")
            return
        CustomFit.best_val_loss = best_val_loss
        CustomFit.best_val_acc = best_val_acc
        print(f"Resuming training from epoch {last_epoch+1}")

    # Setup optimizer and custom model
    optimizer = keras.optimizers.Adam(learning_rate=cfg.HP_LR)
    custom_model = CustomFit(model, acc_metric)
    custom_model.compile(optimizer=optimizer, loss=loss)



    # Training loop
    for epoch in range(last_epoch,cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}:")

        # Train the model
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(total=train_dataloader_length, colour='white', desc=f"Train ", leave=False, unit="steps", ncols=100)
        for batch_index, batch in enumerate(train_dataloader):
            train_results = custom_model.train_step(batch)
            total_loss += train_results["loss"].numpy()
            total_accuracy += train_results["accuracy"].numpy()
            num_batches += 1

            mean_train_loss = total_loss / num_batches
            mean_train_accuracy = total_accuracy / num_batches

            # Update progress bar and display metrics
            progress_bar.update(1)
            progress_bar.set_postfix(loss=mean_train_loss, accuracy=mean_train_accuracy)

        progress_bar.close()

        # Validation step
        val_losses = 0
        val_accuracies = 0
        num_batches = 0
        progress_bar = tqdm(total=val_dataloader_length, colour='white', desc=f" Val ", leave=False, unit="steps", ncols=100)
        for batch_index, batch in enumerate(val_dataloader):
            val_results = custom_model.test_step(batch)
            val_losses += val_results["loss"].numpy()
            val_accuracies += val_results["accuracy"].numpy()
            num_batches += 1

            mean_val_loss = val_losses / num_batches
            mean_val_accuracy = val_accuracies / num_batches

            # Update progress bar and display metrics
            progress_bar.update(1)
            progress_bar.set_postfix(loss=mean_val_loss, accuracy=mean_val_accuracy)

        progress_bar.close()

        # Print Epoch results
        print(f"Train loss: {mean_train_loss:.3f}, Accuracy: {mean_train_accuracy:.3f} | Validation loss: {mean_val_loss:.3f}, Accuracy: {mean_val_accuracy:.3f}")
        log_to_file(log_file_path, epoch, mean_train_loss, mean_train_accuracy, mean_val_loss, mean_val_accuracy)

        # Save the last model weights at the end of each epoch
        custom_model.model.save_weights(run_dir+"/last_model_weights.weights.h5")
        # Save the model if the validation loss improved
        custom_model.save_if_best(mean_val_loss, mean_val_accuracy, filepath=run_dir+"/best_model_weights.weights.h5")

        # Check for early stopping
        if mean_val_loss < best_val_loss or mean_val_accuracy < best_val_acc:
            best_val_loss = mean_val_loss
            best_val_acc = mean_val_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= cfg.patience:
            print(f"No improvement in validation for {cfg.patience} consecutive epochs. Stopping training.")
            break

    # Load the best model and evaluate on test data
    custom_model.model.load_weights(run_dir+"/best_model_weights.weights.h5")

    test_losses = 0
    test_accuracies = 0
    num_batches = 0
    progress_bar = tqdm(total=test_dataloader_length, colour='white', desc=f" Test ", leave=False, unit="steps", ncols=100)
    for batch_index, batch in enumerate(test_dataloader):
        test_results = custom_model.test_step(batch)
        test_losses += test_results["loss"].numpy()
        test_accuracies += test_results["accuracy"].numpy()
        num_batches += 1

        mean_test_loss = test_losses / num_batches
        mean_test_accuracy = test_accuracies / num_batches

        # Update progress bar and display metrics
        progress_bar.update(1)
        progress_bar.set_postfix(loss=mean_test_loss, accuracy=mean_test_accuracy)

    progress_bar.close()

    print(f"Test loss: {mean_test_loss:.3f}, Accuracy: {mean_test_accuracy:.3f}")

    # Delete the last model weights:
    os.remove(run_dir+"/last_model_weights.weights.h5")

