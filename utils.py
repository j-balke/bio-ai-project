from itertools import product

def get_combinations(**hyperparameters):
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    
    return [dict(zip(keys, combination)) for combination in product(*values)]

def get_save_path(config: dict, hyperparameter: dict) -> str:
    pass

def mask_config(config: dict) -> dict:
    #mask token key
    masked_config = config.copy()
    masked_config.pop("token")
    return masked_config

def get_model_path(config: dict) -> str:
    return f"./assets/models/{config['dataset']}/{config['model']}_{config['num_unfreezed_layers']}_unfreezed_layers"
