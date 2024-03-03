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
    load_models
)

def test(config, diffusion_extractor, aggregation_network, files_list):
    device = config.get("device", "cuda")
    output_size, load_size = get_rescale_size(config)
    pck_threshold = config["pck_threshold"]
    test_dist_all, test_pck_img_all, test_pck_bbox_all = [], [], []
    test_pck_img_perimg, test_pck_bbox_perimg = [], []

    test_dist_category_dict = {"aeroplane":[],"bicycle":[],"bird":[],"boat":[],"bottle":[],"bus":[],"car":[],"cat":[],"chair":[],"cow":[],"dog":[],"horse":[],"motorbike":[],"person":[],"pottedplant":[],"sheep":[],"train":[],"tvmonitor":[]}
    test_pck_img_category_dict = {"aeroplane":[],"bicycle":[],"bird":[],"boat":[],"bottle":[],"bus":[],"car":[],"cat":[],"chair":[],"cow":[],"dog":[],"horse":[],"motorbike":[],"person":[],"pottedplant":[],"sheep":[],"train":[],"tvmonitor":[]}
    test_pck_bbox_category_dict = {"aeroplane":[],"bicycle":[],"bird":[],"boat":[],"bottle":[],"bus":[],"car":[],"cat":[],"chair":[],"cow":[],"dog":[],"horse":[],"motorbike":[],"person":[],"pottedplant":[],"sheep":[],"train":[],"tvmonitor":[]}
    
    test_pck_img_category_perimg_dict = {"aeroplane":[],"bicycle":[],"bird":[],"boat":[],"bottle":[],"bus":[],"car":[],"cat":[],"chair":[],"cow":[],"dog":[],"horse":[],"motorbike":[],"person":[],"pottedplant":[],"sheep":[],"train":[],"tvmonitor":[]}
    test_pck_bbox_category_perimg_dict = {"aeroplane":[],"bicycle":[],"bird":[],"boat":[],"bottle":[],"bus":[],"car":[],"cat":[],"chair":[],"cow":[],"dog":[],"horse":[],"motorbike":[],"person":[],"pottedplant":[],"sheep":[],"train":[],"tvmonitor":[]}


    current_category = None
    category_count = 0
    
    #test_anns = json.load(open(config["test_path"]))
    for j, an in tqdm(enumerate(files_list)):
        with torch.no_grad():
            ann = json.load(open(os.path.join(config["test_path"],an)))
            source_points, target_points, img1_pil, img2_pil, imgs = load_image_pair_for_test(ann, load_size, device, image_path=config["image_path"])
            img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            # Assuming compute_clip_loss is used for testing as well
            #loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            #wandb.log({"test/loss": loss.item()}, step=j)
            
            # Log NN correspondences
            _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
            predicted_points = predicted_points.detach().cpu().numpy()
            
            # Rescale to the original image dimensions
            target_size = ann["trg_imsize"]
            predicted_points = rescale_points(predicted_points, load_size, target_size)
            target_points = rescale_points(target_points, load_size, target_size)
            
            dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
            _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["trg_bndbox"])
            
            wandb.log({"test/sample_pck_img": sample_pck_img}, step=j)
            wandb.log({"test/sample_pck_bbox": sample_pck_bbox}, step=j)
            print(ann["category"])
            
            test_dist_all.append(dist)
            test_pck_img_all.append(pck_img)
            test_pck_bbox_all.append(pck_bbox)
            test_pck_img_perimg.append(sample_pck_img)
            test_pck_bbox_perimg.append(sample_pck_bbox)

            test_dist_category_dict[ann["category"]].append(dist)
            test_pck_img_category_dict[ann["category"]].append(pck_img)
            test_pck_bbox_category_dict[ann["category"]].append(pck_bbox)
            test_pck_img_category_perimg_dict[ann["category"]].append(sample_pck_img)
            test_pck_bbox_category_perimg_dict[ann["category"]].append(sample_pck_bbox)
    
        if j % 100 ==0 and j > 0:
            test_pck_img_wandb = np.concatenate(test_pck_img_all)
            test_pck_bbox_wandb = np.concatenate(test_pck_bbox_all)
    
            wandb.log({"test/pck_img": test_pck_img_wandb.sum() / len(test_pck_img_wandb)})
            wandb.log({"test/pck_bbox": test_pck_bbox_wandb.sum() / len(test_pck_bbox_wandb)})

            wandb.log({"test/pck_img_perimage": np.mean(test_pck_img_perimg)})
            wandb.log({"test/pck_bbox_perimage": np.mean(test_pck_bbox_perimg)})

        
        if current_category is None:
            current_category = ann["category"]
        elif current_category != ann["category"]:
            test_pck_img_wandb_category = np.concatenate(test_pck_img_category_dict[current_category])
            test_pck_bbox_wandb_category = np.concatenate(test_pck_bbox_category_dict[current_category])
    
            wandb.log({f"test/{current_category}/pck_img": test_pck_img_wandb_category.sum() / len(test_pck_img_wandb_category)})
            wandb.log({f"test/{current_category}/pck_bbox": test_pck_bbox_wandb_category.sum() / len(test_pck_bbox_wandb_category)})

            wandb.log({f"test/{current_category}/pck_img_perimage": np.mean(test_pck_img_category_perimg_dict[current_category])})
            wandb.log({f"test/{current_category}/pck_bbox_perimage": np.mean(test_pck_bbox_category_perimg_dict[current_category])})
            current_category = ann["category"]
            category_count = 0
        elif category_count % 100 ==0 and category_count > 0:
            test_pck_img_wandb_category = np.concatenate(test_pck_img_category_dict[current_category])
            test_pck_bbox_wandb_category = np.concatenate(test_pck_bbox_category_dict[current_category])
    
            wandb.log({f"test/{current_category}/pck_img": test_pck_img_wandb_category.sum() / len(test_pck_img_wandb_category)})
            wandb.log({f"test/{current_category}/pck_bbox": test_pck_bbox_wandb_category.sum() / len(test_pck_bbox_wandb_category)})

            wandb.log({f"test/{current_category}/pck_img_perimage": np.mean(test_pck_img_category_perimg_dict[current_category])})
            wandb.log({f"test/{current_category}/pck_bbox_perimage": np.mean(test_pck_bbox_category_perimg_dict[current_category])})

        category_count += 1
    
    test_dist_all = np.concatenate(test_dist_all)
    test_pck_img_all = np.concatenate(test_pck_img_all)
    test_pck_bbox_all = np.concatenate(test_pck_bbox_all)
    df = pd.DataFrame({
        "distances": test_dist_all,
        "pck_img": test_pck_img_all,
        "pck_bbox": test_pck_bbox_all,
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
    
    aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])
    test(config, diffusion_extractor, aggregation_network, files_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/test.yaml")
    args = parser.parse_args()
    main(args)