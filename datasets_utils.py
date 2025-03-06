from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch
import os
from kagglehub import dataset_download
import shutil
from sklearn.model_selection import train_test_split
import glob
from config import get_config


class OxfordPetDataset(Dataset):
    def __init__(self, set_type, config, hyperparameter=None):
        set_type = "training" if set_type == "train" else set_type
        self.path = f"{config['oxford_pet_path']}1/{set_type}_set/{set_type}_set/"
        self.set_type = set_type
        self.data_augmentation_prob = hyperparameter["data_augmentation_prob"] if hyperparameter is not None else 0.0

        dirs = [dir for dir in os.listdir(self.path)]

        self.files, self.labels = [], []
        for dir in dirs:
            assert dir in ["cats", "dogs"]
            for file in os.listdir(self.path + dir):
                if not file.endswith(".jpg"):
                    continue
                self.files.append(f"{dir}/{file}")
                label = 1 if dir == "cats" else 0
                self.labels.append(label)

        self.transform = self.get_transform(self.data_augmentation_prob, self.set_type)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.path + self.files[idx]
        img = Image.open(path)
        img = self.transform(img)
        label = torch.tensor(self.labels[idx])
        return (img, label, path)
    
    def get_transform(self, data_augmentation_prob: float, set_type: str):
        transform_list = []
        transform_list.append(transforms.Resize((224, 224)))

        if data_augmentation_prob > 0.0 and self.set_type == "train":
            transform_list.append(transforms.RandomHorizontalFlip(p=data_augmentation_prob))
            transform_list.append(transforms.RandomVerticalFlip(p=data_augmentation_prob))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)
    
    def get_num_classes(self):
        return len(set(self.labels))
    

class BreakHisDataset(Dataset):
    def __init__(self, set_type, config, hyperparameter=None, fold=1):
        self.df = pd.read_csv(f"{config['breakhis_path']}4/Folds_{set_type}_{config['mag']}.csv").reset_index(drop=True)
        self.path = f"{config['breakhis_path']}4/BreaKHis_v1/"
        self.set_type = set_type
        self.data_augmentation_prob = hyperparameter["data_augmentation_prob"] if hyperparameter is not None else 0.0

        self.transform = self.get_transform(self.data_augmentation_prob, self.set_type)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        path = self.path + self.df.loc[idx, "filename"]
        img = Image.open(path)
        img = self.transform(img)
        label = torch.tensor(self.df.loc[idx, "label"])
        return (img, label, path)

    def get_transform(self, data_augmentation_prob: float, set_type: str):
        transform_list = []
        transform_list.append(transforms.Resize((224, 224)))

        if data_augmentation_prob > 0.0 and self.set_type == "train":
            transform_list.append(transforms.RandomHorizontalFlip(p=data_augmentation_prob))
            transform_list.append(transforms.RandomVerticalFlip(p=data_augmentation_prob))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)
        
    def get_num_classes(self):
        return len(set(self.df["label"]))
    
class MultiCancerDataset(Dataset):
    def __init__(self, data_df, set_type, config, hyperparameter=None):
        self.df = data_df.reset_index(drop=True)
        self.set_type = set_type
        self.data_augmentation_prob = hyperparameter["data_augmentation_prob"] if hyperparameter is not None else 0.0
        self.transform = self.get_transform(self.data_augmentation_prob, self.set_type)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        path = self.df.loc[idx, "filename"]
        img = Image.open(path)
        img = self.transform(img)
        label = torch.tensor(self.df.loc[idx, "label"])
        return (img, label, path)

    def get_transform(self, data_augmentation_prob: float, set_type: str):
        transform_list = []
        transform_list.append(transforms.Resize((224, 224)))

        if data_augmentation_prob > 0.0 and self.set_type == "train":
            transform_list.append(transforms.RandomHorizontalFlip(p=data_augmentation_prob))
            transform_list.append(transforms.RandomVerticalFlip(p=data_augmentation_prob))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)
        
    def get_num_classes(self):
        return len(set(self.df["label"]))
    
class BreastCancerDataset(Dataset):
    def __init__(self, data_df, set_type, config, hyperparameter=None):
        self.df = data_df.reset_index(drop=True)
        self.set_type = set_type
        self.data_augmentation_prob = hyperparameter["data_augmentation_prob"] if hyperparameter is not None else 0.0
        self.transform = self.get_transform(self.data_augmentation_prob, self.set_type)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        path = self.df.loc[idx, "filename"]
        img = Image.open(path)
        img = self.transform(img)
        label = torch.tensor(self.df.loc[idx, "label"])
        return (img, label, path)

    def get_transform(self, data_augmentation_prob: float, set_type: str):
        transform_list = []
        transform_list.append(transforms.Resize((224, 224)))

        if data_augmentation_prob > 0.0 and self.set_type == "train":
            transform_list.append(transforms.RandomHorizontalFlip(p=data_augmentation_prob))
            transform_list.append(transforms.RandomVerticalFlip(p=data_augmentation_prob))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)
        
    def get_num_classes(self):
        return len(set(self.df["label"]))
    
def load_oxford_pet(config, hyperparameter):
    """
    Function to download and load Oxford Pet Dataset
    """
    if not os.path.exists(config["oxford_pet_path"]):
        os.makedirs(config["oxford_pet_path"])
        path = dataset_download("tongpython/cat-and-dog")
        shutil.move(path, config["oxford_pet_path"])
        
    train_data = OxfordPetDataset("train", config, hyperparameter)
    val_data = OxfordPetDataset("test", config, hyperparameter)
    test_data = OxfordPetDataset("test", config, hyperparameter)
    return train_data, val_data, test_data

def load_breakhis(config, hyperparameter, small=False):
    """
    Function to download and load BreakHis Dataset
    """
    if not os.path.exists(config["breakhis_path"]):
        os.makedirs(config["breakhis_path"])
        path = dataset_download("ambarish/breakhis")
        shutil.move(path, config["breakhis_path"])

    if not os.path.exists(f"{config['breakhis_path']}4/Folds_{config['mag']}_train.csv"):
        df = pd.read_csv(f"{config['breakhis_path']}4/Folds.csv")
        df = df[df["fold"] == 1]
        df = df[df["mag"] == config["mag"]]

        df["label"] = df["filename"].apply(lambda path: 1 if path.split("/")[3] == "malignant" else 0)
        
        train_df = df[df["grp"] == "train"]
        test_df = df[df["grp"] == "test"]

        train_df, val_df = train_test_split(train_df, test_size=config["val_size"], stratify=train_df["label"], random_state=config["seed"])
        if small:
            assert config["dataset"] == "breakhis_small"
            train_df, _ = train_test_split(train_df, test_size=config["reduction_value"], stratify=train_df["label"], random_state=config["seed"])

        train_df.to_csv(f"{config['breakhis_path']}4/Folds_train_{config['mag']}.csv", index=False)
        val_df.to_csv(f"{config['breakhis_path']}4/Folds_val_{config['mag']}.csv", index=False)
        test_df.to_csv(f"{config['breakhis_path']}4/Folds_test_{config['mag']}.csv", index=False)

    train_data = BreakHisDataset("train", config, hyperparameter)
    val_data = BreakHisDataset("val", config, hyperparameter)
    test_data = BreakHisDataset("test", config, hyperparameter)
    return train_data, val_data, test_data

def load_breast_cancer(config, hyperparameter):
    if not os.path.exists(config["breast_cancer_path"]):
        os.makedirs(config["breast_cancer_path"])
        path = dataset_download("andrewmvd/breast-cancer-cell-segmentation")
        shutil.move(path, config["breast_cancer_path"])

    pathes = os.listdir(f"{config['breast_cancer_path']}1/Images")
    pathes = [f"{config['breast_cancer_path']}1/Images/{path}" 
            for path in pathes if path.endswith(".tif")]

    labels = [1 if "malignant" in path else 0 for path in pathes]

    df = pd.DataFrame({"filename": pathes, "label": labels})

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=config["seed"])
    train_df, val_df = train_test_split(train_df, test_size=config["val_size"], stratify=train_df["label"], random_state=config["seed"])

    train_data = BreastCancerDataset(data_df=train_df, set_type="train", config=config, hyperparameter=hyperparameter)
    val_data = BreastCancerDataset(data_df=val_df, set_type="val", config=config, hyperparameter=hyperparameter)
    test_data = BreastCancerDataset(data_df=test_df, set_type="test", config=config, hyperparameter=hyperparameter)

    return train_data, val_data, test_data
    

def load_multi_cancer(config, hyperparameter, small=False):

    if not os.path.exists(config["multi_cancer_path"]):
        os.makedirs(config["multi_cancer_path"])
        path = dataset_download("obulisainaren/multi-cancer")
        shutil.move(path, config["multi_cancer_path"])

    cancer_type = config["multi_cancer_type"]
    if cancer_type is None:
        cancer_type = "Lymphoma"
    classes = glob.glob(f"{config['multi_cancer_path']}/3/Multi Cancer/Multi Cancer/{cancer_type}/*/")
    data_df = pd.DataFrame(columns=["filename", "label"])
    for label, dir_path in enumerate(classes):
        files = glob.glob(dir_path + "/*")
        data_df = pd.concat([data_df, pd.DataFrame({"filename": files, "label": label})])

    train_df, test_df = train_test_split(data_df, test_size=0.2, stratify=data_df["label"], random_state=config["seed"])
    train_df, val_df = train_test_split(train_df, test_size=config["val_size"], stratify=train_df["label"], random_state=config["seed"])

    if small:
        assert config["dataset"] == "multi_cancer_small"
        train_df, _ = train_test_split(train_df, test_size=config["reduction_value"], stratify=train_df["label"], random_state=config["seed"])


    train_data = MultiCancerDataset(data_df=train_df, set_type="train", config=config, hyperparameter=hyperparameter)
    val_data = MultiCancerDataset(data_df=val_df, set_type="val", config=config, hyperparameter=hyperparameter)
    test_data = MultiCancerDataset(data_df=test_df, set_type="test", config=config, hyperparameter=hyperparameter)

    return train_data, val_data, test_data

def get_data_loader(config, hyperparameter=None):
    """
    Downloads Dataset given in the config dict if not present and returns DataLoaders
    """
    assert config["dataset"] in ["oxford_pet", "breakhis", "multi_cancer", "breakhis_small", "multi_cancer_small", "breast_cancer"]
    
    if config["dataset"] == "oxford_pet":
        train_data, val_data, test_data = load_oxford_pet(config, hyperparameter)
        

    if config["dataset"] == "breakhis":
        train_data, val_data, test_data = load_breakhis(config, hyperparameter)

    if config["dataset"] == "multi_cancer":
        train_data, val_data, test_data = load_multi_cancer(config, hyperparameter)

    if config["dataset"] == "breast_cancer":
        train_data, val_data, test_data = load_breast_cancer(config, hyperparameter)

    if config["dataset"] == "breakhis_small":
        train_data, val_data, test_data = load_breakhis(config, hyperparameter, small=True)
    
    if config["dataset"] == "multi_cancer_small":
        train_data, val_data, test_data = load_multi_cancer(config, hyperparameter, small=True)

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_samples(data_loader, num_per_class=30):
    """
    Function to get test samples from the DataLoader for each class
    """
    dataset = data_loader.dataset
    num_classes = dataset.get_num_classes()
    samples = []

    # for i in range(len(dataset)):
    #     input, label, path = dataset[i]
    #     samples.append((input, label, path))

    for i in range(num_classes):
        class_samples = []
        for j in range(len(dataset)):
            input, label, path = dataset[j]
            if label == i:
                class_samples.append((input, label, path))
                if len(class_samples) == num_per_class:
                    break

        samples.extend(class_samples)

    return samples


if __name__ == "__main__":
    config = get_config("uni", "breakhis")
    a,b, c = get_data_loader(config)



    

        


        
