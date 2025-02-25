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
    def __init__(self, input_dim, num_classes, hidden_layer_dims, dropout=0.1):
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

def load_model(config: dict, hyperparamter: dict, num_classes: int) -> nn.Module:
    """
    Load model from huggingface hub and add classifier head to it
    """
    # login(token=config["token"])

    assert config["model"] in ["uni", "vit", "resnet", "conch"], "Error: Model not supported"
    
    if config["model"] == "uni":
        os.makedirs(config["uni_path"], exist_ok=True)
        # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=config["uni_path"], force_download=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(f"{config['uni_path']}pytorch_model.bin", map_location="cpu"), strict=True)
    elif config["model"] == "vit":
        model = timm.create_model('vit_large_patch16_224.augreg_in21k', pretrained=True, checkpoint_path=config["vit_path"] + "model.pth")
    elif config["model"] == "resnet":
        model = timm.create_model('resnet50.a1_in1k', pretrained=True, checkpoint_path=config["resnet_path"] + "model.pth")
        model
    elif config["model"] == "conch":
        # hf_hub_download("MahmoodLab/CONCH", filename="pytorch_model.bin", local_dir=config["conch_path"], force_download=True)
        model = create_model_from_pretrained('conch_ViT-B-16', config["conch_path"] + "pytorch_model.bin", return_transform=False ) 
    
    head = ClassifierHead(model.embed_dim, num_classes, hyperparamter["layer_dims"], hyperparamter["dropout"])
    model.head = head
    return model

def freeze_layers(model, num_unfreezed_layers: int) -> None:
    """
    freeze all layers of given model except for head and norm layers before head
    unfreeze last num_unfreezed_layers blocks of the model
    """
    num_blocks = len(model.blocks)
    for name, param in model.named_parameters():
        if name.startswith(tuple(["head", "norm"])):
            param.requires_grad = True
            continue

        param.requires_grad = False

    if num_unfreezed_layers == 0:
        return
    
    for i in range(num_blocks - num_unfreezed_layers, num_blocks):
        block = model.blocks[i]
        for param in block.parameters():
            param.requires_grad = True

    return

def evaluate(config, model, dataloader) -> dict:
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

def train(config: dict, hyperparameter: dict, save_best: bool) -> None:
    
    train_data, val_data, test_data = datasets_utils.get_data_loader(config, hyperparameter)
    model = load_model(config, hyperparameter, num_classes=train_data.dataset.get_num_classes()).to(config["device"])

    # print(f"freeze layers: {config['num_unfreezed_layers']}")
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
        
        # print(f"Epoch: {epoch}, Loss: {loss.item()}")

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
    
    if test_data:
        test_scores = evaluate(config, model, test_data)
        print(f"Test Scores: {test_scores}")

    # remove mode from gpu and free memory to avoid out of memory error
    del model
    torch.cuda.empty_cache()

    return {"accuracy": best_acc, "f1_score": best_f1, "recall": best_recall, "precision": best_precision}


def grid_search(config: dict, hyperparam_dict: dict) -> None:
    hyperparameters = utils.get_combinations(**hyperparam_dict)
    print(f"Amount of configurations: {len(hyperparameters)}")
    best_hyperparameters = None
    best_f1 = 0
    all_scores = {}

    for hyperparameter in tqdm(hyperparameters, "grid_search configurations"):
        hyperparameter["epochs"] = 1
        scores = train(config, hyperparameter, save_best=False)

        all_scores[str(hyperparameter)] = scores

        if scores[config["metric"]] > best_f1:
            best_f1 = scores[config["metric"]]
            best_hyperparameters = hyperparameter

    return best_hyperparameters, all_scores

def train_best_model(config: dict, best_hyperparameters: dict) -> nn.Module:
    best_hyperparameters["epochs"] = 8
    scores = train(config, best_hyperparameters, save_best=True)

    return scores

            
if __name__ == "__main__":
    conf = config.get_config("uni", "breakhis")
    
    model = load_model(conf, {"layer_dims": [256], "dropout": 0.1}, 2)
    # torch.save(model.state_dict(), conf["vit_path"]+ "model.pth")