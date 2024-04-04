import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from omegaconf import OmegaConf
import pandas as pd
import torch
from tqdm import tqdm
import wandb
from PIL import Image
from torch.nn import functional as F
import torch.nn as nn

from utils.util import (
    load_image_pair_for_test,
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    draw_correspondences,
    compute_pck,
    rescale_points,
    process_image
)
from utils.wandb_util import get_wandb_run_name
from src.detectron2.resnet import collect_dims
from src.aggregation_network import AggregationNetwork
from src.feature_extractor import CDHFExtractor
from train_aggregation_network import (
    get_rescale_size, 
    get_hyperfeats,
    compute_clip_loss,
    load_models,
    get_feats,
    get_multifeats
)

def test(config, diffusion_extractor, aggregation_network, files_list, args, device="cuda"):
    splits = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
    output_size, load_size = get_rescale_size(config)
    pck_threshold = config["pck_threshold"]

    dataset_path = args.dataset_path
    test_path = 'PairAnnotation/test'
    json_list = os.listdir(os.path.join(dataset_path, test_path))
    all_cats = os.listdir(os.path.join(dataset_path, 'JPEGImages'))
    cat2json = {}

    for cat in all_cats:
        cat_list = []
        for i in json_list:
            if cat in i:
                cat_list.append(i)
        cat2json[cat] = cat_list
    
    cat2img = {}
    for cat in all_cats:
        cat2img[cat] = []
        cat_list = cat2json[cat]
        for json_path in cat_list:
            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)
                temp_f.close()
            src_imname = data['src_imname']
            trg_imname = data['trg_imname']
            if src_imname not in cat2img[cat]:
                cat2img[cat].append(src_imname)
            if trg_imname not in cat2img[cat]:
                cat2img[cat].append(trg_imname)


    print("saving all test images' features...")
    os.makedirs(args.save_path, exist_ok=True)
    for cat in tqdm(all_cats):
        output_dict = {}
        image_list = cat2img[cat]
        for image_path in image_list:
            img_pil = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, image_path))
            img, img1_pil = process_image(img_pil, res=load_size)
            img.to(torch.float16).to(device)
            feats = diffusion_extractor.forward(img)
            # feats_split = torch.split(feats, splits, dim=2)
            output_dict[image_path] = feats.squeeze()
            # output_dict[image_path] = feats_split[3]
        torch.save(output_dict, os.path.join(args.save_path, f'{cat}_LCM_DINO_layer_4_timestep_47.pth'))

    total_pck = []
    all_correct = 0
    all_total = 0

    for cat in all_cats:
        cat_list = cat2json[cat]
        output_dict = torch.load(os.path.join(args.save_path, f'{cat}_LCM_DINO_layer_4_timestep_47.pth'))

        cat_pck = []
        cat_correct = 0
        cat_total = 0

        for json_path in tqdm(cat_list):

            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)

            src_img_size = data['src_imsize'][:2][::-1]
            trg_img_size = data['trg_imsize'][:2][::-1]

            src_ft = output_dict[data['src_imname']]
            trg_ft = output_dict[data['trg_imname']]

            src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft.unsqueeze(0))
            trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft.unsqueeze(0))
            h = trg_ft.shape[-2]
            w = trg_ft.shape[-1]

            trg_bndbox = data['trg_bndbox']
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            total = 0
            correct = 0

            for idx in range(len(data['src_kps'])):
                total += 1
                cat_total += 1
                all_total += 1
                src_point = data['src_kps'][idx]
                trg_point = data['trg_kps'][idx]

                num_channel = src_ft.size(1)
                src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1) # HW, C
                src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1
                trg_vec = F.normalize(trg_vec) # HW, c
                cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy() # H, W

                max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

                dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                if (dist / threshold) <= 0.1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1

            cat_pck.append(correct / total)
        total_pck.extend(cat_pck)

        print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
        print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
        wandb.log({f"test/{cat}/per_image_PCK_0.1": np.mean(cat_pck) * 100})
        wandb.log({f"test/{cat}/per_point_PCK_0.1": cat_correct / cat_total * 100})
    print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
    print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')
    wandb.log({"test/All_per_image_PCK_0.1": np.mean(total_pck) * 100})
    wandb.log({"test/All_per_point_PCK_0.1": all_correct / all_total * 100})

def main(args):
    device="cuda:1"
    config, diffusion_extractor, aggregation_network = load_models(args.config_path, device=device)
    wandb.init(project=config["wandb_project"],name=f"zero_shot_new_metric_{get_wandb_run_name(config)}__layer_4_timestep_47")
    wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]}
    ]
    
    #optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    
    #val_anns = json.load(open(config["val_path"]))
    #test_anns = json.load(open(config["test_path"]))  # Path to your test data JSON
    files_list = os.listdir(config["test_path"])
    
    # aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])
    test(config, diffusion_extractor, aggregation_network, files_list, args, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="path to config yaml", default="configs/test_new_metric.yaml")
    parser.add_argument("--dataset_path", type=str, help="path to spair dataset", default="/home/zzw5373/wh/datasets/SPair-71k/")
    parser.add_argument("--save_path", type=str, help="path to save features", default="/home/zzw5373/wh/ConsistencyDifFeature/spair_ft/")
    args = parser.parse_args()
    main(args)