import logging
import os
import datetime
import config
import argparse

import model_utils
import utils

# Create a directory for logs if it doesn't exist
os.makedirs("logs", exist_ok=True)   

# Set up argument parsing for the script
parser = argparse.ArgumentParser(description='Biomedicine Foundation Model Project')
parser.add_argument('--model', type=str, choices=['uni', 'conch', 'vit', 'resnet'], default='uni',
                    help='Choose the model to for training (default: fedavg)')
args = parser.parse_args()
model = args.model

# Get the list of datasets and grid search parameters from the config
datasets = config.get_datasets()
GRID_PARAMS = config.get_grid_params()

def train_model_with_dataset(model: str, dataset: str, GRID_PARAMS: dict, CONFIG: dict) -> None:
    """
    Trains a model on a specific dataset using grid search to find the best hyperparameters.

    Args:
        model (str): The name of the model to train.
        dataset (str): The name of the dataset to train on.
        GRID_PARAMS (dict): A dictionary of hyperparameters for grid search.
        CONFIG (dict): Configuration dictionary containing model and dataset settings.
    """
    # Set up logging for the training process
    logging.basicConfig(filename=f'logs/{CONFIG["dataset"]}_{CONFIG["model"]}_freeze_{CONFIG["num_unfreezed_layers"]}.log', level=logging.INFO, filemode='w')
    logging.info(f"Training model {model} with dataset {dataset}")
    logging.info(f"TIMESTAMP: {datetime.datetime.now()}")
    logging.info(f"CONFIG: {utils.mask_config(CONFIG)}")
    logging.info(f"GRID_PARAM: {GRID_PARAMS}")

    # Perform grid search to find the best hyperparameters
    best_hyperparameter, grid_scores = model_utils.grid_search(CONFIG, GRID_PARAMS)
    logging.info(f"BEST HYPERPARAMETER: {best_hyperparameter}")
    logging.info(f"GRID SCORES: {grid_scores}")
    
    # Train the model with the best hyperparameters
    scores = model_utils.train_best_model(CONFIG, best_hyperparameter)
    logging.info(f"SCORES: {scores}")

# either freeze the entire base model, or train the last four layers of the base model
# Iterate over all datasets and train the model
for dataset in datasets:
    # Determine the number of layers to unfreeze based on the model type
    if model == "resnet":
        freeze_list = [0]       # ResNet models do not support unfreezing layers
    else:
        freeze_list = [0, 4]    # Unfreeze 0 or 4 layers for other models
    
    # Train the model with different numbers of unfrozen layers
    for num_unfreeze_layers in freeze_list:
        CONFIG = config.get_config(model, dataset)
        CONFIG["num_unfreezed_layers"] = num_unfreeze_layers

        # GRID_PARAM = GRID_PARAMS[model] currently same for all but may change in future
        # Train the model with the current configuration
        train_model_with_dataset(model, dataset, GRID_PARAMS, CONFIG)