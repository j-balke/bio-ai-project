
import logging
import os
import datetime
import config
import argparse

import model_utils
import utils
# todo argparse einbauen

model = 1
os.makedirs("logs", exist_ok=True)   


parser = argparse.ArgumentParser(description='Biomedicine Foundation Model Project')
parser.add_argument('--model', type=str, choices=['uni', 'conch', 'vit', 'resnet'], default='uni',
                    help='Choose the model to for training (default: fedavg)')
args = parser.parse_args()
model = args.model

datasets = config.get_datasets()
GRID_PARAMS = config.get_grid_params()

def train_model_with_dataset(model: str, dataset: str, GRID_PARAMS: dict, CONFIG: dict) -> None:
    logging.info(f"Training model {model} with dataset {dataset}")
    logging.basicConfig(filename=f'logs/{CONFIG["dataset"]}_{CONFIG["model"]}_freeze_{CONFIG["num_unfreezed_layers"]}.log', level=logging.INFO, filemode='w')
    logging.info(f"TIMESTAMP: {datetime.datetime.now()}")
    logging.info(f"CONFIG: {utils.mask_config(CONFIG)}")
    logging.info(f"GRID_PARAM: {GRID_PARAMS}")

    best_hyperparameter, grid_scores = model_utils.grid_search(CONFIG, GRID_PARAMS)
    logging.info(f"BEST HYPERPARAMETER: {best_hyperparameter}")
    logging.info(f"GRID SCORES: {grid_scores}")
    scores = model_utils.train_best_model(CONFIG, best_hyperparameter)
    logging.info(f"SCORES: {scores}")

# either freeze the entire base model, or train the last four layers of the base model
for num_freeze_layers in [0, 4]:
    for dataset in datasets:
        CONFIG = config.get_config(model, dataset)
        CONFIG["num_unfreezed_layers"] = num_freeze_layers

        # GRID_PARAM = GRID_PARAMS[model] currently same for all but may change in future
        train_model_with_dataset(model, dataset, GRID_PARAMS, CONFIG)
