import argparse
from Train import train_model
import importlib.util

def main():
    import ROSS.cfg.ROSS_Config as cfg
    # Create the parser
    parser = argparse.ArgumentParser(description='Train the ROSS model.')

    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--remove_bad_sequences', type=bool, help='Whether to remove bad sequences')
    parser.add_argument('--radar_range', type=int, help='Radar range')
    parser.add_argument('--ROSS_FOV', type=int, help='ROSS Field of View °')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--GT_mode', type=int, help='0: All Data, 1: Only obstacles in range')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')

    # Parse the arguments
    args = parser.parse_args()

    if args.config_path != cfg.config_Path and args.config_path is not None:
        # Update the cfg module
        spec = importlib.util.spec_from_file_location("cfg", args.config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)

    # Dictionary to update config values
    updates = {
        'Remove_bad_sequences': args.remove_bad_sequences,
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

    if 'Experience_' in args.config_path and any(value is not None for value in updates.values()):
        raise ValueError("Cannot update configuration values of a saved experiment. Please create a new experiment.")

    # Call the training function
    train_model(cfg, args.config_path)

if __name__ == '__main__':
    main()