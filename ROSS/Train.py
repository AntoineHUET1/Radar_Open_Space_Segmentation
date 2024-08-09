import os
import tensorflow as tf
from tensorflow import keras
from ROSS.Utils import create_dataloader, genrerat_Graph
from ROSS.Model import build_ROSS_32_50
from ROSS.Model import Metric_MAE,CustomLoss
from time import sleep
from tqdm import tqdm
import warnings



# Filter out specific TensorFlow warnings
warnings.filterwarnings("ignore", message="TF-TRT Warning")

Save_path='/home/watercooledmt/PycharmProjects/ROS/ROS/'

#Set seed:

# ==================== Parameters ====================

# Train Test Val:
Val_ratio=0.1
Test_ratio=0.1

# Input and Output shape:
GT_Output_shape=(32, 2)
input_shape=(256, 256, 1)

# Sequence path:
Data_path='/home/watercooledmt/PycharmProjects/ROS/Datasets/Pixset/'

# Remove bad sequences:
Remove_bad_sequences=False
Bad_sequences=['20200730_003948_part44_2995_3195','20200730_003948_part44_5818_6095','20200730_003948_part44_6875_7500','20200803_151243_part45_1028_1128','20200803_151243_part45_1260_1524','20200803_151243_part45_2310_2560','20200803_151243_part45_4780_5005','20200803_174859_part46_1108_1219','20200803_174859_part46_2761_2861','20200805_002607_part48_2083_2282']
 
# 50 m or 25 m detection:
Half_length= True

Radar_Range=25 # 50,25,20,15,10,5 

#Merge t-1,t and t+1 Radar images:
Merge_Radar_images=1#0,1,2 # 0: No, Oui, Oui IMU

if Merge_Radar_images==2:
    Bad_Sequences= ['20200730_003948_part44_2995_3195', '20200730_003948_part44_2995_3195', '20200706_144800_part25_2160_2784', '20200706_164938_part20_3225_3810','20200622_142617_part18_450_910']
    Remove_bad_sequences=True

# Categorical or regression:
Mode=1 # 0: Regression, 1: Categorical

#Binary Camera:
Binary_Camera=False

FOV=90 # 0: 120, 1: 90

# Have GT data:
GT_mode=1 # 0: All Data, 1: Only obstacles in range

num_epochs = 100
patience = 10  # Number of epochs to wait for improvement

HP_NUM_UNITS = [16]#[32, 64, 128]
HP_DROPOUT = [0.2]#[0.1, 0.2, 0.3, 0.5]
HP_LR = [1e-5]#[1e-3, 1e-4, 1e-5]

Add_Frontal_images=0 #0: No, 1:  + Resized, 2:  +Bird_view, 3: +Bird_view_RA, 4: Only Resized, 5: Only Bird_view, 6: Only Bird_view_RA

Type_of_Input = ['Radar',' Radar_Frontal','Radar_Bird_view','Radar_Bird_view_RA','Frontal','Bird_view','Bird_view_RA']



# ==================== Callbacks ====================

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=patience,
            verbose=0,
            mode="auto",
            min_lr=0.000001,
        )
    ]

# ==================== Set Up Data ====================
#(None, 250, 585, 3)  (None, 250, 585, 4)  
if Half_length:
    input_shape = (128, 256, 1)

if Add_Frontal_images!=0:
    if Add_Frontal_images in [1,2,3]:
        input_shape = (250, 585, 4)
    else:
        input_shape = (250, 585, 3)

# Test, Train, Validation:
Sequence_List=os.listdir(Data_path)

# Remove bad sequences from list:
if Remove_bad_sequences:
    for bad_sequence in Bad_sequences:
        Sequence_List.remove(bad_sequence)
# Remove bad sequences from list:

Number_of_files=len(Sequence_List)
Val_files=int(Number_of_files*Val_ratio)
Test_files=int(Number_of_files*Test_ratio)
Train_files=Number_of_files-Val_files-Test_files


Train_sequence_paths=[]
Val_sequence_paths=[]
Test_sequence_paths=[]

for i in range(Train_files):
    Train_sequence_paths.append(Data_path+Sequence_List[i])
for i in range(Train_files,Train_files+Val_files):
    Val_sequence_paths.append(Data_path+Sequence_List[i])
for i in range(Train_files+Val_files,Number_of_files):
    Test_sequence_paths.append(Data_path+Sequence_List[i])

'''
Easy = ['20200706_202209_part31_2636_2746' , '20200610_185206_part1_5095_5195']
Medium = ['20200616_151155_part9_750_900','20200615_184724_part6_5180_5280']
Hard = ['20200706_162218_part21_790_960','20200622_142945_part19_480_700' , '20200615_184724_part6_5180_5280']
Night = ['20200730_003948_part44_2995_3195','20200730_003948_part44_5818_6095','20200730_003948_part44_6875_7500']
Rain = ['20200803_151243_part45_1028_1128','20200803_151243_part45_2310_2560']

Train_sequence_paths=['/home/watercooledmt/PycharmProjects/ROS/Datasets/Pixset/'+ x for x in Easy]
Val_sequence_paths=['/home/watercooledmt/PycharmProjects/ROS/Datasets/Pixset/'+ x for x in Night]
Test_sequence_paths=['/home/watercooledmt/PycharmProjects/ROS/Datasets/Pixset/'+ x for x in Rain]
'''
#7h45 
# ==================== Visualize data ====================
# Raw data:
#visualize_data(Data_path+'20200706_143808_part26_1200_1360')

# Modified data:
#visualize_modified_data([Data_path+'20200618_175654_part15_1380_1905'],GT_Output_shape)

# ==================== Dataloader ====================

def Generate_Data(batch_size=32,FOV=120):
    train_dataloader, train_dataloader_length = create_dataloader(Train_sequence_paths, input_shape, GT_Output_shape, batch_size, Half_length, GT_mode,Mode,Add_Frontal_images,FOV,Merge_Radar_images,Binary_Camera,Radar_Range)
    val_dataloader , val_dataloader_length = create_dataloader(Val_sequence_paths,input_shape,GT_Output_shape, batch_size,Half_length,GT_mode,Mode,Add_Frontal_images,FOV,Merge_Radar_images,Binary_Camera,Radar_Range)
    test_dataloader , test_dataloader_length = create_dataloader(Test_sequence_paths,input_shape,GT_Output_shape, batch_size,Half_length,GT_mode,Mode,Add_Frontal_images,FOV,Merge_Radar_images,Binary_Camera,Radar_Range)
    train_dataloader_length,val_dataloader_length,test_dataloader_length=2*523,2*37,2*45
    return train_dataloader, train_dataloader_length, val_dataloader, val_dataloader_length, test_dataloader, test_dataloader_length
# ==================== Test ====================

if Mode==0:
    loss_func=CustomLoss()
else:
    loss_func = CustomLoss()

class CustomFit(keras.Model):
    def __init__(self, model):
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
            print(f"Validation loss improved => Saving model")# "#from {self.best_val_loss:.3f} to {val_loss:.3f} => Saving model.")
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.model.save_weights(filepath)
        else:
            print(f"Validation loss did not improve") # from {self.best_val_loss}.")


# ==================== Model ====================

acc_metric = Metric_MAE(Mode=Mode,Half_length=Half_length,GT_mode=GT_mode)
loss=loss_func
import numpy as np

def train_model(units,drop_rate,learning_rate):

    # Generate data
    train_dataloader, train_dataloader_length, val_dataloader, val_dataloader_length, test_dataloader, test_dataloader_length = Generate_Data(batch_size=units, FOV=FOV)

    checkpoint_path='/home/watercooledmt/PycharmProjects/ROS/ROS/Results/25m/Categorical/Radar/16units_0.2dropout_1e-05learning_rate_1GT_mode_FOV_90_Camera_Binary_False_Radar_Range_25/Experience_2/best_model_weights_3.717.weights.h5'

    genrerat_Graph(checkpoint_path, test_dataloader, GT_Output_shape, Type_of_Input,label='test',Save_fig=True,Binary_Camera=Binary_Camera,Radar_Range=Radar_Range)
    #genrerat_Graph(checkpoint_path, train_dataloader, GT_Output_shape, Type_of_Input, label='train', Save_fig=True,Binary_Camera=Binary_Camera,Radar_Range=Radar_Range)
    genrerat_Graph(checkpoint_path, val_dataloader, GT_Output_shape, Type_of_Input, label='val', Save_fig=True,Binary_Camera=Binary_Camera,Radar_Range=Radar_Range)
    breakpoint()
    #checkpoint_path='/home/watercooledmt/PycharmProjects/ROS/ROS/Results/25m/Categorical/Bird_view_RA/16units_0.2dropout_1e-05learning_rate_1GT_mode_FOV_90_Camera_Binary_True/Experience_1/best_model_weights_0.260.weights.h5'
    #genrerat_Graph(checkpoint_path, test_dataloader, GT_Output_shape, Type_of_Input,label='Rain',Save_fig=True,Binary_Camera=Binary_Camera)#'Medium'
    #genrerat_Graph(checkpoint_path, train_dataloader, GT_Output_shape, Type_of_Input, label='Easy', Save_fig=True,Binary_Camera=Binary_Camera)#'Easy'
    #genrerat_Graph(checkpoint_path, val_dataloader, GT_Output_shape, Type_of_Input, label='Night', Save_fig=True,Binary_Camera=Binary_Camera)#'Hard'
    #checkpoint_path='/home/watercooledmt/PycharmProjects/ROS/ROS/Results/25m/Categorical/Bird_view_RA/16units_0.2dropout_1e-05learning_rate_0GT_mode_FOV_90/Experience_1/best_model_weights_2.372.weights.h5'
    #genrerat_Graph(checkpoint_path, test_dataloader, GT_Output_shape, Type_of_Input,label='Medium',Save_fig=True)
    #genrerat_Graph(checkpoint_path, train_dataloader, GT_Output_shape, Type_of_Input, label='Easy', Save_fig=True)
    #genrerat_Graph(checkpoint_path, val_dataloader, GT_Output_shape, Type_of_Input, label='Hard', Save_fig=True)

    # Create model
    #if Add_Frontal_images!=0:

        #model =build_ROS_32_50_Concatenate(input_shape=input_shape, Mode=Mode, Dropout=drop_rate,Binary_Camera=Binary_Camera)
    #else:
    model = build_ROSS_32_50(input_shape=input_shape, Half_length=Half_length, Mode=Mode, Dropout=drop_rate)

    #model.load_weights('/home/antoine/Code/ROS/ROS/best_model_weights.h5')
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    custom_model = CustomFit(model)
    custom_model.compile(optimizer=optimizer, loss=loss)

    best_val_loss = float('inf')
    best_val_acc = float('inf')

    no_improvement_count = 0

    # Training loop
    for epoch in range(num_epochs):
        if 'progress_bar' in locals():
            progress_bar.close()

        sleep(1)
        print(f"Epoch {epoch + 1}/{num_epochs}:")

        # Train the model
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(total=train_dataloader_length,colour='white',desc=f"Train ",leave=False,unit="steps",ncols=100)
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

        progress_bar = tqdm(total=val_dataloader_length, colour='white', desc=f" Val ",leave=False, unit="steps", ncols=100)

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
        print(f"Train loss: {mean_train_loss:.3f} , MAE: {mean_train_accuracy:.3f} | Validation loss: {mean_val_loss:.3f} , MAE: {mean_val_accuracy:.3f}")
        # Save the model if the validation loss improved
        custom_model.save_if_best(mean_val_loss, mean_val_accuracy,filepath="best_model_weights.weights.h5")

        # Check for early stopping

        if mean_val_loss < best_val_loss :
            best_val_loss = mean_val_loss

            no_improvement_count = 0

        if mean_val_accuracy < best_val_acc :
            best_val_acc = mean_val_accuracy
            no_improvement_count = 0

        if mean_val_loss > best_val_loss: # and mean_val_accuracy > best_val_acc:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"No improvement in validation loss for {patience} consecutive epochs. Stopping training.")
            break
    # When training is over, load the best model and evaluate on test data
    custom_model.model.load_weights("best_model_weights.weights.h5")

    # Evaluate the model on test data
    test_losses = 0
    test_accuracies = 0
    num_batches = 0

    progress_bar = tqdm(total=test_dataloader_length, colour='white', desc=f" Test ",leave=False, unit="steps", ncols=100)

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

    # Model_Save_Dir:

    if Mode == 0:
        mode_prefix = 'Regression'
    else:
        mode_prefix = 'Categorical'

    if Half_length:
        distance_prefix = '25m'
    else:
        distance_prefix = '50m'


    run_dir = Save_path+f"Results/{distance_prefix}/{mode_prefix}/{Type_of_Input[Add_Frontal_images]}/{units}units_{drop_rate}dropout_{learning_rate}learning_rate_{GT_mode}GT_mode_FOV_{FOV}_Camera_Binary_{Binary_Camera}_Radar_Range_{Radar_Range}"


    run_dir=run_dir
    print(run_dir)
    if not os.path.exists(run_dir):
        print("Creating directory")
        os.makedirs(run_dir)
    
    # Create Experience i directory inside run_dir 
    Existing_Experiences=os.listdir(run_dir)
    Experience_number=len(Existing_Experiences)+1
    run_dir=run_dir+f'/Experience_{Experience_number}'
    if not os.path.exists(run_dir):
        print("Creating directory")
        os.makedirs(run_dir)

    checkpoint_path=run_dir + f"/best_model_weights_{mean_test_accuracy:.3f}.weights.h5"
    print(checkpoint_path)
    # Copy best_model_weights.h5 to run_dir
    os.rename("best_model_weights.weights.h5",checkpoint_path )

    # Generate graph:
    genrerat_Graph(checkpoint_path, test_dataloader, GT_Output_shape, Type_of_Input,label='test',Save_fig=True,Binary_Camera=Binary_Camera,Radar_Range=Radar_Range)
    genrerat_Graph(checkpoint_path, train_dataloader, GT_Output_shape, Type_of_Input, label='train', Save_fig=True,Binary_Camera=Binary_Camera,Radar_Range=Radar_Range)
    genrerat_Graph(checkpoint_path, val_dataloader, GT_Output_shape, Type_of_Input, label='val', Save_fig=True,Binary_Camera=Binary_Camera,Radar_Range=Radar_Range)


for units in HP_NUM_UNITS:
    for drop_rate in HP_DROPOUT:
        for learning_rate in HP_LR:
            train_model(units,drop_rate,learning_rate)

            



