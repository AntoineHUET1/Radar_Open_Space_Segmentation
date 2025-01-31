from ROSS.Utils import genrerat_Graph, Generate_Data
import importlib.util
import os

def Test(args):


    # ============ Load the configuration file ============
    spec = importlib.util.spec_from_file_location("cfg", args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    Folder_path = os.path.dirname(args.config_path)

    checkpoint_path = Folder_path + '/best_model_weights.weights.h5'

    # ==================== Set Up Data ====================
    Sequence_List = os.listdir(cfg.Data_path)

    # Remove bad sequences from list:
    if cfg.Remove_bad_sequences:
        Sequence_List = [seq for seq in Sequence_List if seq not in cfg.Bad_sequences]

    if args.label in ['test','train','val']:

        # Test, Train, Validation split:
        Number_of_files = len(Sequence_List)
        Val_files = int(Number_of_files * cfg.Val_ratio)
        Test_files = int(Number_of_files * cfg.Test_ratio)
        Train_files = Number_of_files - Val_files - Test_files

        Train_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[:Train_files]]
        Val_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[Train_files:Train_files + Val_files]]
        Test_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[Train_files + Val_files:]]

        # ============ generate the data with graphs ============
        if args.label == 'train':
            train_dataloader, train_dataloader_length = Generate_Data(cfg, Train_sequence_paths)
            genrerat_Graph(checkpoint_path, train_dataloader, cfg, label='train', Save_fig=True, Show_fig=args.show_graph)

        if args.label == 'val':
            val_dataloader, val_dataloader_length = Generate_Data(cfg, Val_sequence_paths)
            genrerat_Graph(checkpoint_path, val_dataloader, cfg, label='val', Save_fig=True, Show_fig=args.show_graph)

        if args.label == 'test':
            test_dataloader, test_dataloader_length = Generate_Data(cfg, Test_sequence_paths)
            genrerat_Graph(checkpoint_path, test_dataloader, cfg, label='test', Save_fig=True, Show_fig=args.show_graph)

    elif args.label in ['Easy', 'Medium', 'Hard', 'Night', 'Rain']:
        Sequence_List = getattr(cfg, args.label)
        # ============ generate the data with graphs ============
        Sequence_List = [cfg.Data_path + seq for seq in Sequence_List]
        test_dataloader, test_dataloader_length = Generate_Data(cfg, Sequence_List)
        genrerat_Graph(checkpoint_path, test_dataloader, cfg, label=args.label, Save_fig=True, Show_fig=args.show_graph)

    elif args.sequence is not None:
        Sequence_List = [cfg.Data_path + args.sequence]
        test_dataloader, test_dataloader_length = Generate_Data(cfg, Sequence_List)

        # Generate prediction:
        Pred_Full, GT_Full, Pred_val_Full=genrerat_Graph(checkpoint_path, test_dataloader, cfg, label=args.sequence, Save_fig=True, Show_fig=args.show_graph,Generate_pred=True)

        return Pred_Full, GT_Full, Pred_val_Full