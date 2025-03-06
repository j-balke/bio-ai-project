from itertools import product

def get_combinations(**hyperparameters):
    """
    Generates all possible combinations of hyperparameters for grid search.
    
    Args:
        **hyperparameters: A dictionary of hyperparameters and their possible values.
    
    Returns:
        list: A list of dictionaries containing all possible combinations of hyperparameters.
    """
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    
    return [dict(zip(keys, combination)) for combination in product(*values)]

def get_save_path(config: dict, hyperparameter: dict) -> str:
    pass

def mask_config(config: dict) -> dict:
    """
    Masks sensitive information in the configuration, such as the token.
    
    Args:
        config (dict): The original configuration.
    
    Returns:
        dict: A copied configuration without sensitive information.
    """
    #mask token key
    masked_config = config.copy()
    masked_config.pop("token")
    return masked_config

def get_model_path(config: dict) -> str:
    """
    Returns the path where the model is saved based on the configuration.
    
    Args:
        config (dict): The model configuration.
    
    Returns:
        str: The path where the model is saved.
    """
    return f"./assets/models/{config['dataset']}/{config['model']}_{config['num_unfreezed_layers']}_unfreezed_layers"
