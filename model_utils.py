import torch
import torch.nn as nn
import timm
import config
from tqdm import tqdm
from huggingface_hub import login, hf_hub_download
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from conch.open_clip_custom import create_model_from_pretrained
import datasets_utils
import utils

class ClassifierHead(nn.Module):
    """
    A custom classifier head that can be added to a base model.
    """
    def __init__(self, input_dim, num_classes, hidden_layer_dims, dropout=0.1):
        """
        Initializes the ClassifierHead.

        Args:
            input_dim (int): The input dimension of the classifier head.
            num_classes (int): The number of output classes.
            hidden_layer_dims (list): A list of dimensions for the hidden layers.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super(ClassifierHead, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class ConchModel(nn.Module):
    """
    A custom model class for the CONCH model, combining a base model and a classifier head.
    """
    def __init__(self, model, head):
        super(ConchModel, self).__init__()
        self.model = model
        self.head = head
    
    def forward(self, x):
        x = self.model.encode_image(x)
        x = self.head(x)
        return x

def load_model(config: dict, hyperparamter: dict, num_classes: int) -> nn.Module:
    """
    Loads a model from the Hugging Face Hub and adds a classifier head to it.

    Args:
        config (dict): Configuration dictionary containing model and dataset settings.
        hyperparameter (dict): Hyperparameters for the model.
        num_classes (int): The number of output classes.

    Returns:
        nn.Module: The loaded model with a classifier head.
    """
    login(token=config["token"])

    assert config["model"] in ["uni", "vit", "resnet", "conch"], "Error: Model not supported"
    
    if config["model"] == "uni":
        os.makedirs(config["uni_path"], exist_ok=True)
        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=config["uni_path"])
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(f"{config['uni_path']}pytorch_model.bin", map_location="cpu"), strict=True)
        n_features = model.embed_dim
    elif config["model"] == "vit":
        model = timm.create_model('vit_large_patch16_224.augreg_in21k', pretrained=True)
        # model.load_state_dict(torch.load(f"{config['vit_path']}model.pth", map_location="cpu"), strict=True)
        n_features = model.embed_dim
    elif config["model"] == "resnet":
        model = timm.create_model('resnet50.a1_in1k', pretrained=True)
        # model.load_state_dict(torch.load(f"{config['resnet_path']}model.pth", map_location="cpu"), strict=True)
        n_features = model.num_features
        
        head = ClassifierHead(n_features, num_classes, hyperparamter["layer_dims"], hyperparamter["dropout"])
        model.fc = head
        return model 
        
    elif config["model"] == "conch":
        hf_hub_download("MahmoodLab/CONCH", filename="pytorch_model.bin", local_dir=config["conch_path"])
        c_model = create_model_from_pretrained('conch_ViT-B-16', config["conch_path"] + "pytorch_model.bin", return_transform=False) 
        n_features = c_model.embed_dim
        head = ClassifierHead(n_features, num_classes, hyperparamter["layer_dims"], hyperparamter["dropout"])
        model = ConchModel(c_model, head)
        return model 
    
    head = ClassifierHead(n_features, num_classes, hyperparamter["layer_dims"], hyperparamter["dropout"])
    model.head = head
    return model

def freeze_layers(model, num_unfreezed_layers: int) -> None:
    """
    Freezes all layers of the given model except for the head and norm layers before the head.
    Unfreezes the last `num_unfreezed_layers` blocks of the model.

    Args:
        model (nn.Module): The model whose layers are to be frozen/unfrozen.
        num_unfreezed_layers (int): The number of blocks to unfreeze.
    """
    for name, param in model.named_parameters():
        if name.startswith(tuple(["head", "norm", "fc"])):
            param.requires_grad = True
            continue

        param.requires_grad = False

    if num_unfreezed_layers == 0:
        return    
    num_blocks = len(model.blocks)
    for i in range(num_blocks - num_unfreezed_layers, num_blocks):
        block = model.blocks[i]
        for param in block.parameters():
            param.requires_grad = True

    return

def evaluate(config, model, dataloader) -> dict:
    """
    Evaluates the model on the given dataloader and returns performance metrics.

    Args:
        config (dict): Configuration dictionary containing model and dataset settings.
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader containing the evaluation data.

    Returns:
        dict: A dictionary containing evaluation metrics (accuracy, F1 score, recall, precision).
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for img, label, _ in dataloader:
            img, label = img.to(config["device"]), label.to(config["device"])
            output = model(img)
            _, pred = torch.max(output, 1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    num_classes = len(set(y_true))
    if num_classes == 2:
        average = "binary"
    elif num_classes > 2:
        average = "macro"
    else:
        print("Error: Number of classes is less than 2")

    f1_score_ = f1_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)

    return {"f1_score": f1_score_, "recall": recall, "precision": precision, "accuracy": accuracy}

def train(config: dict, hyperparameter: dict, save_best: bool) -> dict:
    """
    Trains the model using the given configuration and hyperparameters.

    Args:
        config (dict): Configuration dictionary containing model and dataset settings.
        hyperparameter (dict): Hyperparameters for training.
        save_best (bool): Whether to save the best model based on validation performance.

    Returns:
        dict: A dictionary containing the best evaluation metrics.
    """
    train_data, val_data, test_data = datasets_utils.get_data_loader(config, hyperparameter)
    model = load_model(config, hyperparameter, num_classes=train_data.dataset.get_num_classes()).to(config["device"])

    if config["model"] == "conch":
        freeze_layers(model.model.visual.trunk, config["num_unfreezed_layers"])
    else:
        freeze_layers(model, config["num_unfreezed_layers"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter["lr"])

    if config["scheduler"]:
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer)

    best_acc, best_f1, best_recall, best_precision = 0, 0, 0, 0    
    for epoch in tqdm(range(config["epochs"]), "training model"):
        model.train()
        for i, (img, label, _) in enumerate(train_data):
            optimizer.zero_grad()
            img, label = img.to(config["device"]), label.to(config["device"])
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        if config["scheduler"]:
            scheduler.step()
        
        evaluation_scores = evaluate(config, model, val_data)
        if evaluation_scores[config["metric"]] > best_f1:
            best_acc = evaluation_scores["accuracy"]
            best_f1 = evaluation_scores["f1_score"]
            best_recall = evaluation_scores["recall"]
            best_precision = evaluation_scores["precision"]

            if save_best:
                path = utils.get_model_path(config)
                os.makedirs(path, exist_ok=True)
                torch.save(model, f"{path}/model.pth")
    
    if save_best:
        path = utils.get_model_path(config)
        model = torch.load(f"{path}/model.pth")
        test_scores = evaluate(config, model, test_data)
        return test_scores

    # remove mode from gpu and free memory to avoid out of memory error
    del model
    torch.cuda.empty_cache()

    return {"accuracy": best_acc, "f1_score": best_f1, "recall": best_recall, "precision": best_precision}


def grid_search(config: dict, hyperparam_dict: dict):
    """
    Performs a grid search over the given hyperparameters to find the best configuration.

    Args:
        config (dict): Configuration dictionary containing model and dataset settings.
        hyperparam_dict (dict): A dictionary of hyperparameters to search over.

    Returns:
        tuple: A tuple containing the best hyperparameters and all scores.
    """
    hyperparameters = utils.get_combinations(**hyperparam_dict)
    print(f"Amount of configurations: {len(hyperparameters)}")
    best_hyperparameters = None
    best_f1 = 0
    all_scores = {}

    for hyperparameter in tqdm(hyperparameters, "grid_search configurations"):
        config["epochs"] = 1
        scores = train(config, hyperparameter, save_best=False)

        all_scores[str(hyperparameter)] = scores

        if scores[config["metric"]] > best_f1:
            best_f1 = scores[config["metric"]]
            best_hyperparameters = hyperparameter

    return best_hyperparameters, all_scores

def train_best_model(config: dict, best_hyperparameters: dict):
    config["epochs"] = 8
    scores = train(config, best_hyperparameters, save_best=True)

    return scores

            
if __name__ == "__main__":
    conf = config.get_config("conch", "breakhis")
    
    model = load_model(conf, {"layer_dims": [256], "dropout": 0.1}, 2)
    print(model)
    # torch.save(model.state_dict(), conf["resnet_path"]+ "model.pth")