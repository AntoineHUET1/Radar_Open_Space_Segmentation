from ROSS.Utils import genrerat_Graph, Generate_Data,prediction
import importlib.util
import os
from ..Model.Model_ROSS import build_ROSS_32_50
import time
def inference(config_path,label=None,):

    # ============ Load the configuration file ============
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    Folder_path = os.path.dirname(config_path)

    checkpoint_path = Folder_path + '/best_model_weights.weights.h5'

    # ==================== Load Model ====================

    model = build_ROSS_32_50(cfg)

    model.load_weights(checkpoint_path)

    # ==================== Set Up Data ====================
    Sequence_List = os.listdir(cfg.Data_path)

    # Remove bad sequences from list:
    if cfg.Remove_bad_sequences:
        Sequence_List = [seq for seq in Sequence_List if seq not in cfg.Bad_sequences]

    if label in ['test','train','val']:

        # Test, Train, Validation split:
        Number_of_files = len(Sequence_List)
        Val_files = int(Number_of_files * cfg.Val_ratio)
        Test_files = int(Number_of_files * cfg.Test_ratio)
        Train_files = Number_of_files - Val_files - Test_files

        Train_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[:Train_files]]
        Val_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[Train_files:Train_files + Val_files]]
        Test_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[Train_files + Val_files:]]


        # ============ generate the data with graphs ============
        if label == 'train':
            Data, _ = Generate_Data(cfg, Train_sequence_paths)

        if label == 'val':
            Data, _ = Generate_Data(cfg, Val_sequence_paths)

        if label == 'test':
            Data, _ = Generate_Data(cfg, Test_sequence_paths)

    else:
        if label in ['Easy', 'Medium', 'Hard', 'Night', 'Rain']:
            Sequence_List = getattr(cfg, label)
            # ============ generate the data with graphs ============
            Sequence_List = [cfg.Data_path + seq for seq in Sequence_List]
            Data, _ = Generate_Data(cfg, Sequence_List)
        else:
            Data, _ = Generate_Data(cfg, [cfg.Data_path + label])

    start = time.time()
    prediction(Data,model,cfg)
    end = time.time()
    print(f'{end - start:.1f}) seconds')

    # Calculate the number of frames per second

    Nb_frames=0
    for Radar, GT in Data:
        for i in range(Radar.shape[0]):
            Nb_frames+=1

    print(f'FPS: {Nb_frames/(end-start):.1f}')

