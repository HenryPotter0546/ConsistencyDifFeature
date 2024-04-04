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
import torch.nn.functional as F

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
from utils.extractor_dino import ViTExtractor

def test(config, diffusion_extractor, aggregation_network, files_list, device="cuda", num_patches=30):
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device=device)
    output_size, load_size = get_rescale_size(config)
    pck_threshold = config["pck_threshold"]
    test_dist, test_pck_img, test_pck_bbox = [], [], []
    splits = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
    
    #test_anns = json.load(open(config["test_path"]))
    for j, an in tqdm(enumerate(files_list)):
        with torch.no_grad():
            ann = json.load(open(os.path.join(config["test_path"],an)))
            source_points, target_points, img1_pil, img2_pil, imgs = load_image_pair_for_test(ann, load_size, device, image_path=config["image_path"])
            del img1_pil, img2_pil
            # img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            # img1_hyperfeats, img2_hyperfeats = get_feats(diffusion_extractor, aggregation_network, imgs)
            feats_split = get_multifeats(diffusion_extractor, aggregation_network, imgs, device=device)


            # extract dinov2 features
            # img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
            # img_batch = (extractor_vit.preprocess_pil(imgs)).cuda()
            features_dino = extractor_vit.extract_descriptors(imgs, layer=11, facet='token')
            features_dino = features_dino.permute(0, 1, 3, 2).reshape(2, -1, num_patches, num_patches)

            # img1_hyperfeats = torch.cat([
            #     feats_split[3][0], 
            #     F.interpolate(feats_split[6][0], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            #     F.interpolate(feats_split[9][0], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            #     features_dino[0].unsqueeze(0)], dim=1)
            # img2_hyperfeats = torch.cat([
            #     feats_split[3][1], 
            #     F.interpolate(feats_split[6][1], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            #     F.interpolate(feats_split[9][1], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            #     features_dino[1].unsqueeze(0)], dim=1)

            img1_hyperfeats = torch.cat([
                feats_split[3][0], 
                feats_split[6][0],
                feats_split[9][0], 
                features_dino[0].unsqueeze(0)], dim=1)
            img2_hyperfeats = torch.cat([
                feats_split[3][1], 
                feats_split[6][1],
                feats_split[9][1], 
                features_dino[1].unsqueeze(0)], dim=1)

            del feats_split, features_dino
            
            # Assuming compute_clip_loss is used for testing as well
            #loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            #wandb.log({"test/loss": loss.item()}, step=j)
            
            # Log NN correspondences
            _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
            del img1_hyperfeats, img2_hyperfeats, source_points

            predicted_points = predicted_points.detach().cpu().numpy()
            
            # Rescale to the original image dimensions
            target_size = ann["trg_imsize"]
            predicted_points = rescale_points(predicted_points, load_size, target_size)
            target_points = rescale_points(target_points, load_size, target_size)
            
            dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
            _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["trg_bndbox"])
            del predicted_points, target_points
            
            wandb.log({"test/sample_pck_img": sample_pck_img}, step=j)
            wandb.log({"test/sample_pck_bbox": sample_pck_bbox}, step=j)
            del sample_pck_img, sample_pck_bbox

            print(ann["category"])
            
            test_dist.append(dist)
            del dist

            test_pck_img.append(pck_img)
            del pck_img

            test_pck_bbox.append(pck_bbox)
            del pck_bbox
    
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
    device = "cuda:2"
    config, diffusion_extractor, aggregation_network = load_models(args.config_path, device=device)
    wandb.init(project=config["wandb_project"],name=f"DINO_LCM_Test_{get_wandb_run_name(config)}")
    wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]}
    ]
    
    #optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    
    #val_anns = json.load(open(config["val_path"]))
    #test_anns = json.load(open(config["test_path"]))  # Path to your test data JSON
    files_list = os.listdir(config["test_path"])
    num_patches = config["num_patches"]
    
    # aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])
    test(config, diffusion_extractor, aggregation_network, files_list, device=device, num_patches=num_patches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/test.yaml")
    args = parser.parse_args()
    main(args)


