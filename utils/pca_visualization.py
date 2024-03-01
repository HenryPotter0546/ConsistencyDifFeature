import argparse
import PIL.Image
import numpy
import torch
from pathlib import Path
from src.feature_extractor import CDHFExtractor
from src.aggregation_network import AggregationNetwork
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from typing import List, Tuple
from omegaconf import OmegaConf
import torchvision
from src.detectron2.resnet import collect_dims
import matplotlib.pyplot as plt
import math
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
from math import sqrt
from torchvision import transforms as T

def pca(
        config, 
        image_paths, 
        load_size: int = 224, 
        layer: int = 11, 
        facet: str = 'key', 
        bin: bool = False, 
        stride: int = 2,
        model_type: str = 'dino_vits8', 
        n_components: int = 4,
        all_together: bool = True
    ) -> List[Tuple[Image.Image, numpy.ndarray]]:
    """
    finding pca of a set of images.
    :param image_paths: a list of paths of all the images.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :param n_components: number of pca components to produce.
    :param all_together: if true apply pca on all images together.
    :return: a list of lists containing an image and its principal components.
    """

    save_path= config.get("save_path")
    weights_path = "/home/zzw5373/ConsistentDiffusionHyperfeatures/top_ckpt/checkpoint_step_2500.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion_extractor = CDHFExtractor(config)
    dims = config.get("dims")
    if dims is None:
        dims = collect_dims(diffusion_extractor.pipe.unet, idxs=diffusion_extractor.idxs)
    if config.get("flip_timesteps", False):
        config["save_timestep"] = config["save_timestep"][::-1]
    aggregation_network = AggregationNetwork(
            projection_dim=config["projection_dim"],
            feature_dims=dims,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
    )
    aggregation_network.load_state_dict(torch.load(weights_path, map_location="cpu")["aggregation_network"])
    descriptors_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []

    # extract descriptors and saliency maps for each image
    for image_path in image_paths:
        #image_batch, image_pil = extractor.preprocess(image_path, load_size)
        #descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls=False).cpu().numpy()
        image_pil = Image.open(f"{image_path}").convert("RGB")
        save_path = '/home/zzw5373/ConsistentDiffusionHyperfeatures/visual_figure'+'/origin.jpg'
        image_pil.save(save_path)
        image, image_pil = process_image(image_pil, res=load_size)
        image_pil_list.append(image_pil)


        descs, feats = get_hyperfeats(diffusion_extractor, aggregation_network, image)
        splits = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        feats_split = torch.split(feats, splits, dim=2)  # feats_split[0].shape [1,2,1280,64,64]
        # b, c, H, W = descs.shape
        # p = 0

        for index, channel in enumerate(splits):
            b = feats_split[index].size(2)
            h = feats_split[index].size(3)
            image = feats_split[index][0,0,:,:,:]

            # apply pca to all channels: c in [1,2,c,h,w]
            image = image.view(b, -1).float()

            # calculate the mean
            s_mean = torch.mean(image, dim=0,keepdim=True)

            # rank_k mean top k main components in PCA
            rank_k = 3

            # SVD
            _,_,VT = torch.pca_lowrank(image-s_mean, q = rank_k) 

            # take top-k main components
            VT_k = VT[:,:rank_k]

            # visualize top-k 
            for i in range(rank_k):
                image = VT_k[:,i].view(h,-1)
                npimg = image.cpu().numpy()
                plt.imshow(npimg, interpolation='nearest')
                plt.axis('off')
                plt.savefig('/home/zzw5373/ConsistentDiffusionHyperfeatures/PCA_result/'+'layer_'+str(index)+'_top'+str(i)+'.jpg')

            # assign top 3 to rgb channels and visualize it.
            image = VT_k[:,:3].view(h,h,3)
            image = (image-image.min())/(image.max()-image.min())
            npimg = image.cpu().numpy()
            plt.imshow(npimg, interpolation='nearest')
            plt.axis('off')
            plt.savefig('/home/zzw5373/ConsistentDiffusionHyperfeatures/PCA_result/'+'layer_'+str(index)+'.jpg')
        
       


def process_image(image_pil, res=None, range=(-1, 1)):
    if res:
        image_pil = image_pil.resize((res, res), Image.BILINEAR)
    image = torchvision.transforms.ToTensor()(image_pil) # range [0, 1]
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min # range [r_min, r_max]
    return image[None, ...], image_pil


def get_hyperfeats(diffusion_extractor, aggregation_network, imgs):
    with torch.inference_mode():
        with torch.autocast("cuda"):
            feats = diffusion_extractor.forward(imgs)
            b, s, l, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)))   # torch.Size([1, 2, 10560, 64, 64]) ->torch.Size([1, 384, 64, 64])
    img_hyperfeats = diffusion_hyperfeats[0][None, ...]
    return img_hyperfeats, feats


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor PCA.')
    parser.add_argument('--root_dir', type=str, default='/home/zzw5373/ConsistentDiffusionHyperfeatures/img/feature', help='The root dir of images.')
    parser.add_argument('--save_dir', type=str, default='PCA_result', help='The root save dir for results.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                    small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                       options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--n_components', default=4, type=int, help="number of pca components to produce.")
    parser.add_argument('--last_components_rgb', default='True', type=str2bool, help="save last components as rgb image.")
    parser.add_argument('--save_resized', default='True', type=str2bool, help="If true save pca in image resolution.")
    parser.add_argument('--all_together', default='False', type=str2bool, help="If true apply pca on all images together.")
    parser.add_argument('--config_path', default='/home/zzw5373/ConsistentDiffusionHyperfeatures/configs/pca.yaml', help="The path to config yaml")

    args = parser.parse_args()

    with torch.no_grad():

        # prepare directories
        config = OmegaConf.load(args.config_path)
        config = OmegaConf.to_container(config, resolve=True)
        root_dir = Path(args.root_dir)
        images_paths = [x for x in root_dir.iterdir() if x.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        save_dir = Path(args.save_dir) 
        save_dir.mkdir(exist_ok=True, parents=True)
        pca_per_image = pca(config, images_paths, args.load_size, args.layer, args.facet, args.bin, args.stride, args.model_type,
                            args.n_components, args.all_together,)

        print("saving images")
        for image_path, (pil_image, pca_image) in tqdm(zip(images_paths, pca_per_image)):
            save_prefix = image_path.stem
            plot_pca(pil_image, pca_image, str(save_dir), args.last_components_rgb, args.save_resized, save_prefix)
