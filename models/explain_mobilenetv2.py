import json
import numpy as np
import torch
import torchvision
import shap
from config import get_config
from model_utils import load_model
from datasets_utils import get_data_loader

config = get_config("uni", "breakhis")
train_loader, val_loader, test_loader = get_data_loader(config)

# Load model
model = load_model(config, {"layer_dims": [256], "dropout": 0.1}, 2).to(config["device"])
model.eval()

# Retrieve ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]

# Data preprocessing
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2) if x.shape[-1] == 3 else x

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1) if x.shape[1] == 3 else x

transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
    torchvision.transforms.Normalize(mean=mean, std=std),
    torchvision.transforms.Lambda(nchw_to_nhwc),
])

inv_transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                                     std=(1 / np.array(std)).tolist()),
    torchvision.transforms.Lambda(nchw_to_nhwc),
])

def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img)).to(config["device"])
    return model(img)

# SHAP explanation model for an image
masker_blur = shap.maskers.Image("blur(128,128)", (224, 224, 3))
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

# Get a sample image from the test loader
sample_img, _, _ = next(iter(test_loader))
shap_values = explainer(sample_img, max_evals=10000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])

# Transform back the results
shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

# Visualization
shap.image_plot(shap_values=shap_values.values, pixel_values=shap_values.data, labels=shap_values.output_names)
