import argparse
from ROSS.Utils.Train import train_model
from ROSS.Utils.Test import Test
from ROSS.Utils.Inference import inference
from ROSS.Utils.Visualize import visualize_data
import importlib.util

def main():
    import ROSS.cfg.ROSS_Config as cfg
    # Create the parser
    parser = argparse.ArgumentParser(description='Train the ROSS model.')

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'visualize','test','inference'],
                        help='Mode to run: train, visualize, or inference')

    # Global arguments
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--sequence', type=str, help='Sequence to visualize')

    # Training arguments
    parser.add_argument('--radar_range', type=int, help='Radar range')
    parser.add_argument('--ROSS_FOV', type=int, help='ROSS Field of View °')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--GT_mode', type=int, help='0: All Data, 1: Only obstacles in range')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')


    # Visualize arguments
    parser.add_argument('--no_camera', action='store_true',help='Frame number to visualize')
    parser.add_argument('--no_radar', action='store_true',help='Visualize radar files')
    parser.add_argument('--no_GT', action='store_true',help='Visualize GT files')
    parser.add_argument('--frame_number', type=int, help='Frame number to visualize')
    parser.add_argument('--fps', type=int,default=10, help='GT Output shape')
    parser.add_argument('--GT_point_cloud',action='store_true', help='To visualize the GT point cloud instead of stixels')

    # Test arguments
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--label', type=str, help='Path to the output')
    parser.add_argument('--show_graph', action='store_true', help='Sequence to visualize')

    # Inference arguments
    parser.add_argument('--data', type=str, help='Path to the data file')

    # Parse the arguments
    args = parser.parse_args()

    # Training mode
    if args.mode == 'train':
        if args.config_path != cfg.config_Path and args.config_path is not None:
            # Update the cfg module
            spec = importlib.util.spec_from_file_location("cfg", args.config_path)
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)

        # Dictionary to update config values
        updates = {
            'Radar_Range': args.radar_range,
            'FOV': args.ROSS_FOV,
            'patience': args.patience,
            'GT_mode': args.GT_mode,
            'num_epochs': args.num_epochs
        }

        # Update cfg with values from updates dictionary if they are not None
        for key, value in updates.items():
            if value is not None:
                setattr(cfg, key, value)

        # Update input shape based on radar range
        if cfg.Radar_Range > 25:
            cfg.input_shape = (256, 256, 1)
        else:
            cfg.input_shape = (128, 256, 1)

        if args.config_path is not None:
            if 'Experience_' in args.config_path and any(value is not None for value in updates.values()):
                raise ValueError("Cannot update configuration values of a saved experiment. Please create a new experiment.")

        # Call the training function
        train_model(cfg, args.config_path)

    elif args.mode == 'visualize':
        visualize_data(args.sequence,args.no_camera,args.no_radar,args.no_GT,args.frame_number,args.fps,args.GT_point_cloud)

    elif args.mode == 'test':
        # if no model path then return error:
        if args.config_path is None:
            raise ValueError('Please provide a model path')

        if args.label is None and args.sequence is None:
            raise ValueError('Please provide a sequence name or a label')
        Test(args.config_path,label=args.label,show_graph=args.show_graph)

    elif args.mode == 'inference':
        # if no model path then return error:
        if args.config_path is None:
            raise ValueError('Please provide a model path')
        if args.label is None and args.sequence is None and args.data is None:
            raise ValueError('Please provide a sequence name a label or a data path')
        inference(args.config_path,label=args.label)

if __name__ == '__main__':
    main()