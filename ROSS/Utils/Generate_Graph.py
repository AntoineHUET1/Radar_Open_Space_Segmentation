import matplotlib.pyplot as plt
import tensorflow as tf
from ..Model.Model_ROSS import build_ROSS_32_50
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


def genrerat_Graph(checkpoint_path, Data, cfg, label, Save_fig=False):

    Fontsize = 20
    # ==================== Default Val ====================
    Thresholds = [0, 0.2, 0.4, 0.6, 0.8]
    colors = [(204, 51, 51), (255, 106, 0), (255, 204, 0), (212, 255, 0), (122, 221, 122)]

    directory_path = os.path.dirname(checkpoint_path)
    # ==================== Parameters ====================
    Factor = cfg.Radar_Range / 50

    # ==================== Load Model ====================

    model = build_ROSS_32_50(cfg)

    model.load_weights(checkpoint_path)

    # ==================== Visualize data ====================

    # Check if the graph as already been generated
    if (os.path.exists(directory_path + '/' + label + '_Error_Distribution.png') and
        os.path.exists(directory_path + '/' + label + '_Prediction_confidence_level_MAE.png')):
        print(label+' graphs already generated')
        return

    Pred_Full = []
    GT_Full = []
    GT_Full_numpy = []
    Pred_val_Full = []
    Pred_val_Full_numpy = []

    for Radar, GT in Data:
        # Predict
        Predictions = model.predict(Radar)

        Predictions_index = Predictions.argmax(axis=-1)

        # Get the value of the argmax
        Predictions_Val = Predictions.max(axis=-1)

        Predictions_index = Predictions_index.reshape(Radar.shape[0], cfg.GT_Output_shape[0], 1)

        Predictions_Val = Predictions_Val.reshape(Radar.shape[0],  cfg.GT_Output_shape[0], 1)

        for i in range(Radar.shape[0]):
            if  cfg.GT_mode == 0:
                Pred_Full.append(Predictions_index[i])
                GT_Full.append(GT[i][:, 1])
                Pred_val_Full.append(Predictions_Val[i])

            else:

                have_target, pos = GT[i][:, 0], GT[i][:, 1]
                mask = tf.cast(have_target, dtype=tf.bool)

                masked_true_values = tf.boolean_mask(pos, mask)
                masked_pred_values = tf.boolean_mask(Predictions_index[i], mask)
                masked_pred_val = tf.boolean_mask(Predictions_Val[i], mask)

                Pred_Full.append(masked_pred_values.numpy().tolist())
                GT_Full.append(masked_true_values.numpy().tolist())
                Pred_val_Full.append(masked_pred_val.numpy().tolist())
                Pred_val_Full_numpy.append(Predictions_index[i])
                # print(GT[i][:, 1].shape)
                GT_Full_numpy.append(GT[i][:, 1])

    # Save the data as np
    # Flatten the lists
    flattened_list_Pred = [item for sublist in Pred_Full for item in sublist]
    flattened_list_GT = [item for sublist in GT_Full for item in sublist]
    flattened_list_Pred_val = [item for sublist in Pred_val_Full for item in sublist]

    # Convert to List:
    list_Pred = [item[0] * Factor for item in flattened_list_Pred]
    list_GT = [(item.numpy().item() if hasattr(item, 'numpy') else item) * Factor for item in flattened_list_GT]
    # list_GT = [item.numpy().item()*Factor for item in flattened_list_GT]
    list_Pred_val = [item[0] for item in flattened_list_Pred_val]

    plt.figure(figsize=(20, 10))
    # Iterate over each threshold

    list_percentage_GT = []
    list_mae = []
    for i, Threshold in enumerate(Thresholds):
        keep = np.where(np.array(list_Pred_val) > Threshold)[0]

        if len(keep) == 0:
            continue
        # Calculate the error
        mae = mean_absolute_error(np.array(list_Pred)[keep], np.array(list_GT)[keep])

        error_values = np.array(list_Pred)[keep] - np.array(list_GT)[keep]

        list_percentage_GT.append(round(100 * len(keep) / len(list_GT), 2))
        list_mae.append(round(mae, 2))

        # Define bins
        bins = np.arange(-25.5, 25.5, 1)

        # Create histogram
        hist, bins = np.histogram(error_values, bins=bins)

        # Calculate percentage
        percentage = (hist / len(list_Pred)) * 100

        # Color

        color = [c / 255 for c in colors[i]]

        # Plot histogram with percentage
        plt.bar(bins[:-1], percentage, width=1, color=color, label=f'P>{Threshold}')

    # Customize labels and title
    plt.xlabel('Error Values (m)', fontsize=Fontsize)
    plt.ylabel('Percentage', fontsize=Fontsize)
    plt.title('Distribution of Error Values (%) for ' + label + ' data', fontsize=1.5 * Fontsize)

    plt.legend()

    # Show plot
    if Save_fig == True:
        plt.savefig(directory_path + '/' + label + '_Error_Distribution.png')
    else:
        plt.show()

    # Data

    # Normalize the data for bar lengths
    percentage_GT_normalized = [percent / 100 for percent in list_percentage_GT]

    '''
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(15, 10))

    for i in range(len(percentage_GT_normalized)):
        ax.barh(i, percentage_GT_normalized[i], color=np.array(colors[i]) / 255, edgecolor='black')
        ax.text(percentage_GT_normalized[i] + 0.01, i, f'{list_mae[i]:.2f}', va='center', color='black',
                fontsize=Fontsize)

    # Customize plot
    ax.set_yticks(range(len(Thresholds)))

    if Thresholds[0] == 0:
        ax.set_yticklabels(['All data' if t == 0 else f'Threshold > {t}' for t in Thresholds], fontsize=Fontsize)
    else:
        ax.set_yticklabels([f'Threshold > {t}' for t in Thresholds], fontsize=Fontsize)

    ax.set_xlabel('% of Data', fontsize=Fontsize)
    ax.set_title(
        'Mean Absolute Error (MAE) for Predictions within Specific Prediction confidence level (Data ' + label + ')',
        fontsize=Fontsize * 0.8)

    # Show plot
    plt.tight_layout()
    
    # Show plot
    if Save_fig == True:
        plt.savefig(directory_path + '/' + label + '_Prediction_confidence_level.png')
    else:
        plt.show()
    '''
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(15, 10))

    for i in range(len(list_mae)):
        ax.barh(i, list_mae[i], color=np.array(colors[i]) / 255, edgecolor='black')
        ax.text(list_mae[i] + 0.01, i, f'{list_mae[i]:.2f}', va='center', color='black', fontsize=Fontsize)

    # Customize plot
    ax.set_yticks(range(len(Thresholds)))

    if Thresholds[0] == 0:
        ax.set_yticklabels(['All data' if t == 0 else f'Threshold > {t}' for t in Thresholds], fontsize=Fontsize)
    else:
        ax.set_yticklabels([f'Threshold > {t}' for t in Thresholds], fontsize=Fontsize)

    ax.set_xlim(0, 4)

    ax.set_xlabel('MAE', fontsize=Fontsize)
    ax.set_title('Mean Absolute Error (MAE) for Predictions within Specific Prediction Confidence Levels (Data ' + label + ')',fontsize=Fontsize * 0.8)

    # Show plot
    plt.tight_layout()

    # Save or show plot
    if Save_fig:
        plt.savefig(directory_path + '/' + label + '_Prediction_confidence_level_MAE.png')
    else:
        plt.show()
