import torch
import cv2
import numpy as np
import os
import sys
sys.path.append("./vit-explain-main")

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from vit_rollout import VITAttentionRollout, rollout
from vit_grad_rollout import VITAttentionGradRollout

import datasets_utils
import model_utils
import utils
import config
import pandas as pd

GRAD_CAM_METHODS = {"grad-cam": GradCAM, "score-cam": ScoreCAM}
GRAD_CAM_METHODS = {"grad-cam": GradCAM}
DATASETS = config.get_datasets()
MODELS = ["vit", "resnet", "uni", "conch"]
SAVE_PATH = "./assets/explanations"
COLUMNS=["model", "dataset", "num_unfreeze_layers", "method", "label", "original_img_path", "mask_path"]

os.makedirs(SAVE_PATH, exist_ok=True)

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_image(path: str, img_size) -> np.ndarray:
    rgb_img = cv2.imread(path)
    rgb_img = cv2.resize(rgb_img, (img_size, img_size))
    rgb_img = np.float32(rgb_img) / 255
    return rgb_img

def load_model(config):
    model_path = utils.get_model_path(config)
    model_path = "model_pet.pth"
    model = torch.load(model_path, map_location=config["device"])
    model.eval()

    # unfreeze weigths for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True

    # this adaption is necessary for attention rollout, since timm updated its implementation
    # we doublechecked that the results are the same as before
    for block in model.blocks:
        block.attn.fused_attn = False
    
    return model

def get_target_layers(model_name, model):
    if model_name == "resnent":
        return [model.layer4[-1]]
    
    return [model.blocks[-1].norm1]


def get_grad_cam_maps(model, samples, config, target_layers):
    data = []

    for cam_name, cam_method in GRAD_CAM_METHODS.items():
        cam = cam_method(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

        for i, (input, label, path) in enumerate(samples):
            input = input.unsqueeze(0).to(config["device"])
            target = [ClassifierOutputTarget(label.item())]
            grayscale_cam = cam(input_tensor=input, targets=target)[0, :]

            rgb_img = get_image(path, config["img_size"])
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)

            save_path = os.path.join(SAVE_PATH, f"{config["model"]}_{config['dataset']}_{config['num_unfreezed_layers']}_{cam_name}_{i}.jpg")
            cv2.imwrite(save_path, cam_image)

            data.append([config["model"], config["dataset"], config["num_unfreezed_layers"], cam_name, label.item(), path, save_path])

    return pd.DataFrame(data, columns=COLUMNS)

def get_attention_rollout_maps(method, model, samples, config):
    data = []
    head_fusions = ["mean", "min", "max"] if method == "attn_rollout" else ["mean"]

    for head_fusion in head_fusions:
        method_name = f"{method}_{head_fusion}"
        for i, (input, label, path) in enumerate(samples):
            input = input.unsqueeze(0).to(config["device"])

            if method == "attn_rollout":
                attn_rollout = VITAttentionRollout(model, head_fusion=head_fusion)
                rollout_img = attn_rollout(input)
            elif method == "attn_grad_rollout":
                attn_rollout = VITAttentionGradRollout(model)
                rollout_img = attn_rollout(input, label.item())
            else:
                raise ValueError(f"Unknown method: {method}")
            
            rollout_img = cv2.resize(rollout_img, (config["img_size"], config["img_size"]))
            
            rgb_img = get_image(path, config["img_size"])
            cam_image = show_cam_on_image(rgb_img, rollout_img)

            save_path = os.path.join(SAVE_PATH, f"{config["model"]}_{config['dataset']}_{config['num_unfreezed_layers']}_attention_rollout_{head_fusion}_{i}.jpg")
            cv2.imwrite(save_path, cam_image)

            data.append([config["model"], config["dataset"], config["num_unfreezed_layers"], method_name, label.item(), path, save_path])

    return pd.DataFrame(data, columns=COLUMNS)

def get_raw_attn_maps(model, samples, config):
    data = []

    for i, (input, label, path) in enumerate(samples):
        input = input.unsqueeze(0).to(config["device"])
        outputs = {}

        def get_attns(name: str):
            def hook_fn(module, input, output):
                outputs[name] = output.detach()

            return hook_fn
        
        model.blocks[-1].attn.q_norm.register_forward_hook(get_attns("Q"))
        model.blocks[-1].attn.k_norm.register_forward_hook(get_attns("K"))

        model(input)

        scale = model.blocks[-1].attn.scale
        outputs["attn"] = (outputs["Q"] @ outputs["K"].transpose(-2, -1)) * scale
        outputs["attn"] = outputs["attn"].softmax(dim=-1)
        outputs["attn"] = outputs["attn"].mean(dim=1).squeeze(0).detach().cpu().numpy()
        outputs["attn"] = outputs["attn"][0, 1:]

        #normalize
        outputs["attn"] = (outputs["attn"] - outputs["attn"].min()) / (outputs["attn"].max() - outputs["attn"].min())

        attn_map = outputs["attn"].reshape(14, 14)
        attn_map = cv2.resize(attn_map, (config["img_size"], config["img_size"]))

        rgb_img = get_image(path, config["img_size"])
        cam_image = show_cam_on_image(rgb_img, attn_map)

        save_path = os.path.join(SAVE_PATH, f"{config["model"]}_{config['dataset']}_{config['num_unfreezed_layers']}_raw_attn_{i}.jpg")
        cv2.imwrite(save_path, cam_image)

        data.append([config["model"], config["dataset"], config["num_unfreezed_layers"], "raw_attn", label.item(), path, save_path])

        model.blocks[-1].attn.q_norm._forward_hooks.clear()
        model.blocks[-1].attn.k_norm._forward_hooks.clear()

    return pd.DataFrame(data, columns=COLUMNS)



if __name__ == "__main__":
    df_masks = pd.DataFrame(columns=["model", "dataset", "num_unfreeze_layers", "method", "label", "original_img_path", "mask_path"])

    for model in MODELS:
        for dataset in DATASETS:
            for num_unfreezed_layers in [0,4]:
                CONFIG = config.get_config(model, dataset)
                CONFIG["num_unfreezed_layers"] = num_unfreezed_layers
                samples = datasets_utils.get_samples(datasets_utils.get_data_loader(CONFIG)[2])

                model = load_model(CONFIG)

                # grad cam
                target_layers = get_target_layers(CONFIG["model"], model)
                df_cam = get_grad_cam_maps(model, samples, CONFIG, target_layers)
                df_explanations = pd.concat([df_explanations, df_cam])

                if model == "resnet":
                    continue
                # attention rollout
                df_rollout = get_attention_rollout_maps("attn_rollout", model, samples, CONFIG)
                df_explanations = pd.concat([df_explanations, df_rollout])

                df_grad_rollout = get_attention_rollout_maps("attn_grad_rollout", model, samples, CONFIG)
                df_explanations = pd.concat([df_explanations, df_grad_rollout])

                # raw attentions
                df_raw_attn = get_raw_attn_maps(model, samples, CONFIG)
                df_explanations = pd.concat([df_explanations, df_raw_attn])

                del model
                torch.cuda.empty_cache()
    
    df_explanations.to_csv(f"{SAVE_PATH}/data.csv", index=False)
