import json
import torch

def get_config(model: str, dataset: str) -> dict:
    config = {
        "dataset": dataset,
        "model": model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 10,
        "metric": "f1_score",
        "epochs": 2,
        "scheduler": False,
        "val_size": 0.1,
        "uni_path": "./assets/models/uni/",
        "resnet_path": "./assets/models/resnet/",
        "vit_path": "./assets/models/vit/",
        "conch_path": "./assets/models/conch/",
        "breakhis_path": "./assets/data/breakhis/",
        "oxford_pet_path": "./assets/data/oxford_pet/",
        "multi_cancer_path": "./assets/data/multi_cancer/",
        "multi_cancer_type": None,
    }

    if dataset in ["oxford_pet", "breakhis"]:
        config["num_classes"] = 2
        config["img_size"] = 224

    if dataset == "breakhis":
        config["mag"] = 400

    with open("./token.json", "r") as f:
        config["token"] = json.load(f)["token"]

    return config

# def get_models() -> list:
#     # vit: https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k
#     # resnet: https://huggingface.co/timm/resnet50.a1_in1k
#     # uni: https://huggingface.co/MahmoodLab/UNI
#     return ["uni", "resnet", "vit", "conch"]

def get_datasets() -> list:
    # return ["breakhis"]
    return ["breakhis", "oxford_pet", "multi_cancer"]

def get_grid_params() -> dict:
    grid_param = dict()
    grid_param["lr"] = [1e-5, 1e-4]
    grid_param["layer_dims"] = [[256], [], [512, 64]]
    grid_param["data_augmentation_prob"] = [0, 0.5]
    grid_param["num_unfreezed_layers"] = [0,2,6]
    grid_param["dropout"] = [0.1]

    return grid_param
