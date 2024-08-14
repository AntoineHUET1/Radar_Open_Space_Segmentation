import os
import tensorflow as tf
from tensorflow import keras
from ROSS.Utils import genrerat_Graph,Generate_Data
from ROSS.Model import build_ROSS_32_50, CustomFit
from ROSS.Model import Metric_MAE, CustomLoss
from time import sleep
from tqdm import tqdm
import warnings
import ROSS.cfg.ROSS_Config as cfg


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
os.chdir(script_dir)

# Filter out specific TensorFlow warnings
warnings.filterwarnings("ignore", message="TF-TRT Warning")

default_config_path = '.ROSS/cfg/ROSS_Config.py'

Save_path = './Results'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)

# ==================== Set Up Data ====================

# List files in the ROSS_Dataset directory
Sequence_List = os.listdir(cfg.Data_path)

# Remove bad sequences from list:
if cfg.Remove_bad_sequences:
    Sequence_List = [seq for seq in Sequence_List if seq not in cfg.Bad_sequences]

# Test, Train, Validation split:
Number_of_files = len(Sequence_List)
Val_files = int(Number_of_files * cfg.Val_ratio)
Test_files = int(Number_of_files * cfg.Test_ratio)
Train_files = Number_of_files - Val_files - Test_files

Train_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[:Train_files]]
Val_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[Train_files:Train_files + Val_files]]
Test_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[Train_files + Val_files:]]

# Radar_Range:
if cfg.Radar_Range<=25:
    input_shape = (128, 256, 1)


# ==================== Model ====================

acc_metric = Metric_MAE(cfg)
loss = CustomLoss()

# ==================== Callbacks ====================

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=cfg.patience,
        verbose=0,
        mode="auto",
        min_lr=0.000001,
    )
]

def train_model(cfg):


    run_dir = Save_path + f"/Radar/{cfg.Radar_Range}m/"

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Create Experience i directory inside run_dir
    Existing_Experiences = os.listdir(run_dir)
    Experience_number = len(Existing_Experiences) + 1
    run_dir = run_dir + f'/Experience_{Experience_number}'
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Save the configuration file in the experience directory
    os.system(f"cp {default_config_path} {run_dir}/config.py")

    # Generate data
    train_dataloader, train_dataloader_length, val_dataloader, val_dataloader_length, test_dataloader, test_dataloader_length = Generate_Data(cfg, Train_sequence_paths, Val_sequence_paths, Test_sequence_paths)

    model = build_ROSS_32_50(cfg)

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=cfg.HP_LR)
    custom_model = CustomFit(model, acc_metric)
    custom_model.compile(optimizer=optimizer, loss=loss)

    best_val_loss = float('inf')
    best_val_acc = float('inf')

    no_improvement_count = 0

    # Training loop
    for epoch in range(cfg.num_epochs):
        if 'progress_bar' in locals():
            progress_bar.close()

        sleep(1)
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}:")

        # Train the model
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(total=train_dataloader_length, colour='white', desc=f"Train ", leave=False, unit="steps",
                            ncols=100)
        for batch_index, batch in enumerate(train_dataloader):
            train_results = custom_model.train_step(batch)
            total_loss += train_results["loss"].numpy()

            x, y = batch
            predict = custom_model.model(x, training=True)
            total_accuracy += train_results["accuracy"].numpy()
            num_batches += 1
            mean_train_loss = total_loss / num_batches
            mean_train_accuracy = total_accuracy / num_batches

            # Update progress bar and display metrics
            progress_bar.update(1)
            progress_bar.set_postfix(loss=mean_train_loss, accuracy=mean_train_accuracy)

        progress_bar.close()

        progress_bar = tqdm(total=val_dataloader_length, colour='white', desc=f" Val ", leave=False, unit="steps",
                            ncols=100)

        # Evaluate the model on validation data
        val_losses = 0
        val_accuracies = 0
        num_batches = 0

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

        # Print Epoch results (Train loss and accuracy, Validation loss and accuracy)
        print(
            f"Train loss: {mean_train_loss:.3f} , MAE: {mean_train_accuracy:.3f} | Validation loss: {mean_val_loss:.3f} , MAE: {mean_val_accuracy:.3f}")
        # Save the model if the validation loss improved
        custom_model.save_if_best(mean_val_loss, mean_val_accuracy, filepath="best_model_weights.weights.h5")

        # Check for early stopping

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss

            no_improvement_count = 0

        if mean_val_accuracy < best_val_acc:
            best_val_acc = mean_val_accuracy
            no_improvement_count = 0

        if mean_val_loss > best_val_loss:  # and mean_val_accuracy > best_val_acc:
            no_improvement_count += 1

        if no_improvement_count >= cfg.patience:
            print(f"No improvement in validation loss for {cfg.patience} consecutive epochs. Stopping training.")
            break
    # When training is over, load the best model and evaluate on test data
    custom_model.model.load_weights("best_model_weights.weights.h5")

    # Evaluate the model on test data
    test_losses = 0
    test_accuracies = 0
    num_batches = 0

    progress_bar = tqdm(total=test_dataloader_length, colour='white', desc=f" Test ", leave=False, unit="steps",
                        ncols=100)

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

    print(f"Test loss: {mean_test_loss:.3f} , MAE: {mean_test_accuracy:.3f}")

    checkpoint_path = run_dir + f"/best_model_weights_{mean_test_accuracy:.3f}.weights.h5"
    # Copy best_model_weights.h5 to run_dir
    os.rename("best_model_weights.weights.h5", checkpoint_path)

    # Generate graph:
    genrerat_Graph(checkpoint_path, test_dataloader, cfg, label='test', Save_fig=True)
    genrerat_Graph(checkpoint_path, train_dataloader, cfg, label='train', Save_fig=True)
    genrerat_Graph(checkpoint_path, val_dataloader, cfg, label='val', Save_fig=True)


train_model(cfg)