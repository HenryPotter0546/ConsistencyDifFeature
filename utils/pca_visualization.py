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

def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, transform_experiments, t, save_dir):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_fit_data)  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(transform_experiments):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(os.path.join(save_dir, f"{experiment}_time_{t}.png"))

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
        # # feats_split[2] torch.Size([1, 2, 1280, 64, 64])
        # transform_experiments=feats_split[2].shape[1]  # transform_experiments = 2
        # feature_maps_fit_data = feats_split[2].detach().to("cpu").squeeze(dim=0).numpy() # (2, 1280, 64, 64)
        # feature_maps_fit_data=feature_maps_fit_data.reshape(transform_experiments, -1)  #(2, 5242880)
        # pca = PCA(n_components=1)
        # pca.fit(feature_maps_fit_data)
        # feature_maps_pca = pca.transform(feature_maps_fit_data)

        # feature_maps_pca = feature_maps_pca.reshape(transform_experiments, -1, 1)
        # pca_img = feature_maps_pca[0]
        # h = w = int(sqrt(pca_img.shape[0]))
        # pca_img = pca_img.reshape(h, w, 1)
        # pca_img_min = pca_img.min(axis=(0, 1))
        # pca_img_max = pca_img.max(axis=(0, 1))
        # pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        # pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        # pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)





         # feature_maps_fit_data = feats_split[2].detach().to("cpu").squeeze(dim=0).view(-1, descs.shape[1]).numpy()
        # pca = PCA(n_components=3) 
        # pca_transformed = pca.fit_transform(feature_maps_fit_data)
        # h, w = descs.shape[2], descs.shape[3]
        # pca_transformed_images = pca_transformed.reshape(h, w, 3)
        # pca_transformed_images=(pca_transformed_images * 255).astype(np.uint8)
        # image = Image.fromarray(pca_transformed_images, mode='RGB')
        # image.save('/home/zzw5373/ConsistentDiffusionHyperfeatures/visual_figure'+'/layer_3.png')

        # feature_maps_fit_data = descs.clone().detach().to("cpu").squeeze(dim=0).view(-1, descs.shape[1]).numpy() 
        # pca = PCA(n_components=3)
        # pca_transformed = pca.fit_transform(feature_maps_fit_data)

        # h, w = descs.shape[2], descs.shape[3]
        # pca_transformed_images = pca_transformed.reshape(h, w, 3)
        # pca_transformed_images=(pca_transformed_images * 255).astype(np.uint8)

        # image = Image.fromarray(pca_transformed_images, mode='RGB')
        # image.save('/home/zzw5373/ConsistentDiffusionHyperfeatures/visual_figure'+'/pca_hyperfeature.png')

        #image.save(f'/home/zzw5373/ConsistentDiffusionHyperfeatures/visual_figure'+'/pca_layer_{i}.png')

        # 将特征图拉平以适应 seaborn 的热图
        #flat_descs_map = descs.clone().detach().to("cpu").view(384, -1).numpy()

        # 使用 seaborn 绘制热图
        # sns.heatmap(flat_descs_map, cmap='viridis')
        # plt.title('Heatmap of Feature Map')

        # 保存热图
        plt.savefig('/home/zzw5373/ConsistentDiffusionHyperfeatures/visual_figure'+'/heatmap.png')

        plt.show()


        # # 假设 feature_map 是你的特征图张量
        # feature_map = torch.randn(1, 384, 64, 64)

        # # 选择一个通道进行可视化
        # channel_to_visualize = 0
        # single_channel_feature = feature_map[0, channel_to_visualize, :, :].detach().numpy()

        # # 显示单个通道的特征图
        # plt.imshow(single_channel_feature, cmap='viridis')
        # plt.colorbar()
        # plt.title(f'Feature Map - Channel {channel_to_visualize}')
        # plt.show()


        num_patches = (1 + (H - p) // stride, 1 + (W - p) // stride)
        num_patches_list.append(num_patches)

        # load_size_list.append(curr_load_size)
        descriptors_list.append(descs)
    if all_together:
        descriptors = np.concatenate(descriptors_list, axis=2)[0, 0]
        pca = PCA(n_components=n_components).fit(descriptors)
        pca_descriptors = pca.transform(descriptors)
        split_idxs = np.array([num_patches[0] * num_patches[1] for num_patches in num_patches_list])
        split_idxs = np.cumsum(split_idxs)
        pca_per_image = np.split(pca_descriptors, split_idxs[:-1], axis=0)
    else:
        pca_per_image = []
        for descriptors in descriptors_list:
            vis_pca(descriptors)
            pca = PCA(n_components=n_components).fit(descriptors[0,0].cpu().numpy())
            pca_descriptors = pca.transform(descriptors[0,0].cpu().numpy())
            pca_per_image.append(pca_descriptors)
    results = []
    for pil_image, img_pca, num_patches in zip(image_pil_list, pca_per_image, num_patches_list):
        reshaped_pca = img_pca.reshape((64, 64, n_components))
        results.append((pil_image, reshaped_pca))
    return results

def process_image(image_pil, res=None, range=(-1, 1)):
    if res:
        image_pil = image_pil.resize((res, res), Image.BILINEAR)
    image = torchvision.transforms.ToTensor()(image_pil) # range [0, 1]
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min # range [r_min, r_max]
    return image[None, ...], image_pil

def vis_pca(feature1):
    # feature1 shape (1,1,3600,768*2)
    # feature2 shape (1,1,3600,768*2)
    num_patches=int(math.sqrt(feature1.shape[2]))
    # pca the concatenated feature to 3 dimensions
    feature1 = feature1.squeeze() # shape (3600,768*2)
    # feature2 = feature2.squeeze() # shape (3600,768*2)
    chennel_dim, W, H = feature1.shape
    # resize back
    h1, w1 = Image.open("/home/zzw5373/ConsistentDiffusionHyperfeatures/img/feature/ref_img.jpg").size
    scale_h1 = h1/num_patches
    scale_w1 = w1/num_patches
    
    scale = scale_w1
    scaled_h = int(h1/scale)
    feature1 = feature1.permute(1, 2, 0)
    feature1_uncropped=feature1[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]
    
    # h2, w2 = Image.open(trg_img_path).size
    # scale_h2 = h2/num_patches
    # scale_w2 = w2/num_patches
    # if scale_h2 > scale_w2:
    #     scale = scale_h2
    #     scaled_w = int(w2/scale)
    #     feature2 = feature2.reshape(num_patches,num_patches,chennel_dim)
    #     feature2_uncropped=feature2[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
    # else:
    #     scale = scale_w2
    #     scaled_h = int(h2/scale)
    #     feature2 = feature2.reshape(num_patches,num_patches,chennel_dim)
    #     feature2_uncropped=feature2[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]

    f1_shape=feature1_uncropped.shape[:2]
    # f2_shape=feature2_uncropped.shape[:2]
    feature1 = feature1_uncropped.reshape(f1_shape[0]*f1_shape[1],chennel_dim)
    # feature2 = feature2_uncropped.reshape(f2_shape[0]*f2_shape[1],chennel_dim)
    n_components=3
    pca = PCA(n_components=n_components)
    # feature1_n_feature2 = torch.cat((feature1,feature2),dim=0) # shape (7200,768*2)
    # feature1_n_feature2 = pca.fit_transform(feature1_n_feature2.cpu().numpy()) # shape (7200,3)
    # feature1 = feature1_n_feature2[:feature1.shape[0],:] # shape (3600,3)
    # feature2 = feature1_n_feature2[feature1.shape[0]:,:] # shape (3600,3)
    feature1 = pca.fit_transform(feature1.cpu().numpy())
    print(feature1.shape)
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    for show_channel in range(n_components):
        # min max normalize the feature map
        feature1[:, show_channel] = (feature1[:, show_channel] - feature1[:, show_channel].min()) / (feature1[:, show_channel].max() - feature1[:, show_channel].min())
        # feature2[:, show_channel] = (feature2[:, show_channel] - feature2[:, show_channel].min()) / (feature2[:, show_channel].max() - feature2[:, show_channel].min())
        feature1_first_channel = feature1[:, show_channel].reshape(f1_shape[0], f1_shape[1])
        # feature2_first_channel = feature2[:, show_channel].reshape(f2_shape[0], f2_shape[1])

        axes[show_channel, 0].imshow(feature1_first_channel)
        axes[show_channel, 0].axis('off')
        # axes[show_channel, 1].imshow(feature2_first_channel)
        # axes[show_channel, 1].axis('off')
        axes[show_channel, 0].set_title('Feature 1 - Channel {}'.format(show_channel + 1), fontsize=14)
        # axes[show_channel, 1].set_title('Feature 2 - Channel {}'.format(show_channel + 1), fontsize=14)


    feature1_resized = feature1[:, :3].reshape(f1_shape[0], f1_shape[1], 3)
    # feature2_resized = feature2[:, :3].reshape(f2_shape[0], f2_shape[1], 3)

    axes[3, 0].imshow(feature1_resized)
    axes[3, 0].axis('off')
    # axes[3, 1].imshow(feature2_resized)
    # axes[3, 1].axis('off')
    axes[3, 0].set_title('Feature 1 - All Channels', fontsize=14)
    # axes[3, 1].set_title('Feature 2 - All Channels', fontsize=14)

    plt.tight_layout()
    plt.show()
    fig.savefig('/home/zzw5373/ConsistentDiffusionHyperfeatures/PCA_result/pca'+'/pca.png', dpi=300)
    a=1


def plot_pca(pil_image: Image.Image, pca_image: numpy.ndarray, save_dir: str, last_components_rgb: bool = True,
             save_resized=True, save_prefix: str = ''):
    """
    finding pca of a set of images.
    :param pil_image: The original PIL image.
    :param pca_image: A numpy tensor containing pca components of the image. HxWxn_components
    :param save_dir: if None than show results.
    :param last_components_rgb: If true save last 3 components as RGB image in addition to each component separately.
    :param save_resized: If true save PCA components resized to original resolution.
    :param save_prefix: optional. prefix to saving
    :return: a list of lists containing an image and its principal components.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    pil_image_path = save_dir / f'{save_prefix}_orig_img.png'
    pil_image.save(pil_image_path)

    n_components = pca_image.shape[2]
    for comp_idx in range(n_components):
        comp = pca_image[:, :, comp_idx]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        comp_file_path = save_dir / f'{save_prefix}_{comp_idx}.png'
        pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
        if save_resized:
            pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
        pca_pil.save(comp_file_path)

    if last_components_rgb:
        comp_idxs = f"{n_components-3}_{n_components-2}_{n_components-1}"
        comp = pca_image[:, :, -3:]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        comp_file_path = save_dir / f'{save_prefix}_{comp_idxs}_rgb.png'
        pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
        if save_resized:
            pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
        pca_pil.save(comp_file_path)

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