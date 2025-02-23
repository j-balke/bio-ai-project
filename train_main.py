
import logging
import os
import datetime
import config

import model_utils
import utils

os.makedirs("logs", exist_ok=True)   


models = config.get_models()
datasets = config.get_datasets()
GRID_PARAMS = config.get_grid_params()

def train_model_with_dataset(model: str, dataset: str, GRID_PARAMS: dict, CONFIG: dict) -> None:
    logging.info(f"Training model {model} with dataset {dataset}")
    logging.basicConfig(filename=f'logs/{CONFIG["dataset"]}_{CONFIG["model"]}.log', level=logging.INFO, filemode='w')
    logging.info(f"TIMESTAMP: {datetime.datetime.now()}")
    logging.info(f"CONFIG: {utils.mask_config(CONFIG)}")
    logging.info(f"GRID_PARAM: {GRID_PARAMS}")

    best_hyperparameter, grid_scores = model_utils.grid_search(CONFIG, GRID_PARAMS)
    logging.info(f"BEST HYPERPARAMETER: {best_hyperparameter}")
    logging.info(f"GRID SCORES: {grid_scores}")
    scores = model_utils.train_best_model(CONFIG, best_hyperparameter)
    logging.info(f"SCORES: {scores}")

for model in models:
    for dataset in datasets:
        CONFIG = utils.get_config(model, dataset)

        # GRID_PARAM = GRID_PARAMS[model] currently same for all but may change in future
        train_model_with_dataset(model, dataset, GRID_PARAMS, CONFIG)
