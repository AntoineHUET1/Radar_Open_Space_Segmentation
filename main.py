import argparse
from Train import train_model
import importlib.util

def main():
    import ROSS.cfg.ROSS_Config as cfg
    # Create the parser
    parser = argparse.ArgumentParser(description='Train the ROSS model.')

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'visualize', 'inference'],
                        help='Mode to run: train, visualize, or inference')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--radar_range', type=int, help='Radar range')
    parser.add_argument('--ROSS_FOV', type=int, help='ROSS Field of View °')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--GT_mode', type=int, help='0: All Data, 1: Only obstacles in range')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')

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
        if args.radar_range > 25:
            cfg.input_shape = (256, 256, 1)
        else:
            cfg.input_shape = (128, 256, 1)

        if args.config_path is not None:
            if 'Experience_' in args.config_path and any(value is not None for value in updates.values()):
                raise ValueError("Cannot update configuration values of a saved experiment. Please create a new experiment.")

        # Call the training function
        train_model(cfg, args.config_path)

    elif args.mode == 'visualize':
        # Placeholder for data visualization function
        print("Data visualization function is not implemented yet.")

    elif args.mode == 'inference':
        # Placeholder for inference function
        print("Inference function is not implemented yet.")

if __name__ == '__main__':
    main()