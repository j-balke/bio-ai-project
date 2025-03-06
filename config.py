import json
import torch

def get_config(model: str, dataset: str) -> dict:
    """
    Returns a configuration dictionary containing settings for the model and dataset.
    
    Args:
        model (str): The name of the model (e.g., 'uni', 'resnet', 'vit', 'conch').
        dataset (str): The name of the dataset (e.g., 'breakhis', 'oxford_pet').
    
    Returns:
        dict: A dictionary with configurations for the model and dataset.
    """
    config = {
        "dataset": dataset,
        "model": model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 10,
        "metric": "f1_score",
        "epochs": 8,
        "scheduler": False,
        "val_size": 0.1,
        "seed": 0,
        "uni_path": "./assets/models/uni/",
        "resnet_path": "./assets/models/resnet/",
        "vit_path": "./assets/models/vit/",
        "conch_path": "./assets/models/conch/",
        "breakhis_path": "./assets/data/breakhis/",
        "breast_cancer_path": "./assets/data/breast_cancer/",
        "oxford_pet_path": "./assets/data/oxford_pet/",
        "multi_cancer_path": "./assets/data/multi_cancer/",
        "multi_cancer_type": None,
        "img_size": 224
    }

    if dataset in ["oxford_pet", "breakhis", "breakhis_small", "breast_cancer"]:
        config["num_classes"] = 2

    if dataset in ["multi_cancer_small", "breakhis_small"]:
        config["reduction_value"] = 0.8

    if dataset in ["breakhis", "breakhis_small"]:
        config["mag"] = 400

    with open("./token.json", "r") as f:
        config["token"] = json.load(f)["token"]

    return config


def get_datasets() -> list:
    """
    Returns a list of all supported datasets.
    
    Returns:
        list: A list of available datasets.
    """
    return ["breakhis_small", "breakhis", "multi_cancer_small", "oxford_pet", "multi_cancer", "breast_cancer"]

def get_grid_params() -> dict:
    """
    Returns a dictionary with hyperparameters for grid search.
    
    Returns:
        dict: A dictionary with hyperparameters for grid search.
    """
    grid_param = dict()
    grid_param["lr"] = [1e-5, 1e-4]
    grid_param["layer_dims"] = [[256], [], [512, 64]]
    grid_param["data_augmentation_prob"] = [0, 0.5]
    grid_param["dropout"] = [0.1]

    return grid_param
