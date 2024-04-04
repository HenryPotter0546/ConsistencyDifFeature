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

from utils.util import (
    load_image_pair_for_test,
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    draw_correspondences,
    compute_pck,
    rescale_points
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

def test(config, diffusion_extractor, aggregation_network, files_list):
    device = "cuda:2"
    output_size, load_size = get_rescale_size(config)
    pck_threshold = config["pck_threshold"]
    test_dist, test_pck_img, test_pck_bbox = [], [], []
    splits = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
    
    #test_anns = json.load(open(config["test_path"]))
    for j, an in tqdm(enumerate(files_list)):
        with torch.no_grad():
            ann = json.load(open(os.path.join(config["test_path"],an)))
            source_points, target_points, img1_pil, img2_pil, imgs = load_image_pair_for_test(ann, load_size, device, image_path=config["image_path"])
            # img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            img_hyperfeats = get_multifeats(diffusion_extractor, aggregation_network, imgs)
            
            best_layers=[]
            sample_pck_imgs=[]
            sample_pck_bboxs=[]
            pck_bboxs =[]
            pck_imgs =[]
            dists =[]
            test_pck_img_perimg, test_pck_bbox_perimg = [], []

            # 初始化最大值和对应的索引
            max_value = None
            max_index = None


            for k, split in enumerate(splits):
            # Assuming compute_clip_loss is used for testing as well
            #loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            #wandb.log({"test/loss": loss.item()}, step=j)
            
                # Log NN correspondences
                img1_hyperfeats=img_hyperfeats[k][0]
                img2_hyperfeats=img_hyperfeats[k][1]

                _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
                predicted_points = predicted_points.detach().cpu().numpy()
            
                # Rescale to the original image dimensions
                target_size = ann["trg_imsize"]
                predicted_points = rescale_points(predicted_points, load_size, target_size)
                target_points = rescale_points(target_points, load_size, target_size)
            
                dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
                _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["trg_bndbox"])

                sample_pck_imgs.append(sample_pck_img)
                sample_pck_bboxs.append(sample_pck_bbox)
                pck_bboxs.append(pck_bbox)
                pck_imgs.append(pck_img)
                dists.append(dists)

                if max_value is None or pck_bbox > max_value:
                    max_value = pck_bbox
                    max_index = k

            sample_pck_img = sample_pck_imgs[max_index]
            sample_pck_bbox = sample_pck_bboxs[max_index]
            dist = dists[max_index]
            pck_img = pck_imgs[max_index]
            pck_bbox = pck_bboxs[max_index]

            print("prefer layer:", max_index+1)
            
            wandb.log({"test/sample_pck_img": sample_pck_img}, step=j)
            wandb.log({"test/sample_pck_bbox": sample_pck_bbox}, step=j)
            print(ann["category"])
            
            test_dist.append(dist)
            test_pck_img.append(pck_img)
            test_pck_bbox.append(pck_bbox)
            test_pck_img_perimg.append(sample_pck_img)
            test_pck_bbox_perimg.append(sample_pck_bbox)
    
        if j % 100 ==0 and j > 0:
            test_pck_img_wandb = np.concatenate(test_pck_img)
            test_pck_bbox_wandb = np.concatenate(test_pck_bbox)
    
            wandb.log({"test/pck_img": test_pck_img_wandb.sum() / len(test_pck_img_wandb)})
            wandb.log({"test/pck_bbox": test_pck_bbox_wandb.sum() / len(test_pck_bbox_wandb)})
    
    test_dist = np.concatenate(test_dist)
    test_pck_img = np.concatenate(test_pck_img)
    test_pck_bbox = np.concatenate(test_pck_bbox)
    df = pd.DataFrame({
        "distances": test_dist,
        "pck_img": test_pck_img,
        "pck_bbox": test_pck_bbox,
    })
    
    wandb.log({f"test/distances_csv": wandb.Table(dataframe=df)})

def main(args):
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)
    wandb.init(project=config["wandb_project"],name=f"checkpoint_Test_{get_wandb_run_name(config)}")
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
    test(config, diffusion_extractor, aggregation_network, files_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/test.yaml")
    args = parser.parse_args()
    main(args)