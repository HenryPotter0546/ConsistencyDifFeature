from src.feature_extractor import CDHFExtractor
from src.pipe_utils import pipe_selector
from src.cdhf_pipelines import MyLCMPipeline
from src.my_unet import MyUNet2DConditionModel
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch
import tqdm
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os


config_path = "/home/zzw5373/wh/ConsistentDiffusionHyperfeatures/configs/train.yaml"
config = OmegaConf.load(config_path)
config = OmegaConf.to_container(config, resolve=True)

device = "cuda"

LCM_MODELS_DICT = { 1: "LCM_Dreamshaper_v7",
                        2: "lcm-lora-sdv1-5", 
                        3: "lcm-sdxl", 
                        4: "lcm-ssd-1b", 
                        5: "lcm-lora-sdxl", 
                        6: "lcm-lora-ssd-1b",
                        7: "stable-diffusion-v1-5",
                        }
    
lcm_model_name = LCM_MODELS_DICT[1]
# image_path = "./img/feature/ref_img.jpg"
image_path = "./img/feature/ref_img.jpg"
image_path = "./img/feature/ref_whitebear_img.jpg"
save_path = "./img/generate/x0_pred/iid_lcm_generation/"
# image_path = "./img/feature/target_img.jpg"
num_inference_steps = 50
# prompt = "A cat sitting on a chair, real photo"
# prompt = "cat"
prompt = " "

seed = 1
seed = np.random.randint(0, 100000)
seed = 10006
generator = torch.manual_seed(seed) # fix the random seed


cdhf_extractor = CDHFExtractor(config)

additional_embeds = cdhf_extractor.pipe.encode_prompt(prompt=prompt,device=device,num_images_per_prompt=1,do_classifier_free_guidance=False)
cdhf_extractor.pipe.scheduler.set_timesteps(num_inference_steps, device)
timesteps = cdhf_extractor.pipe.scheduler.timesteps


image_tensors = cdhf_extractor.image2tensor(image_path) # image_tensor shape [1, 3, 512, 512]
img_latent = cdhf_extractor.pipe.vae.encode(image_tensors).latent_dist.sample(generator=generator) * cdhf_extractor.pipe.vae.config.scaling_factor # img_latent shape [1, 4, 64, 64]

print(cdhf_extractor.pipe.scheduler)
# t = torch.tensor(19, dtype=torch.long, device=device)

# timesteps = timesteps.flip(dims=[0])
print("init_timtsteps", timesteps)
"""[999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739,
        719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459,
        439, 419, 399, 379, 359, 339, 319, 299, 279, 259, 239, 219, 199, 179,
        159, 139, 119,  99,  79,  59,  39,  19],"""

correct_generation = False
update_ref = False
timesteps.to(device) 

for i, t in tqdm.tqdm(enumerate(timesteps)):
    print("t", t)
    sampled_noise = torch.randn_like(img_latent).to(device)
    x_t_latent = cdhf_extractor.pipe.scheduler.add_noise(img_latent, sampled_noise, t) # we can only add noise in the latent space
    x_t_latent = img_latent

    output = cdhf_extractor.pipe.lcm_generation_one_step(x_t_latent, t, -1, additional_embeds=additional_embeds)
    # x_t_latent = output["x_t_next_latent"]
    x_t_next_latent = output["x_t_next_latent"]
    x_0_predicted = output["x_0_predicted_latent"]

    output_latent = x_0_predicted
    output_latent = cdhf_extractor.pipe.vae.decode(output_latent / cdhf_extractor.pipe.vae.config.scaling_factor, return_dict=False)[0].detach().clone()
    
    x_t_next_output = x_t_next_latent
    x_t_next_output = cdhf_extractor.pipe.vae.decode(x_t_next_output / cdhf_extractor.pipe.vae.config.scaling_factor, return_dict=False)[0].detach().clone()

    # Postprocess image
    images = cdhf_extractor.pipe.image_processor.postprocess(output_latent, output_type="pil")
    x_t_next_images = cdhf_extractor.pipe.image_processor.postprocess(x_t_next_output, output_type="pil")

    if not os.path.exists(f"./img/generate/ref_whitebear/no_iterate_x0_generate/seed_{seed}"):
        os.makedirs(f"./img/generate/ref_whitebear/no_iterate_x0_generate/seed_{seed}")
    images[0].save(f"./img/generate/ref_whitebear/no_iterate_x0_generate/seed_{seed}/no_iterate_x0_generate_timestep{t}.png")

    if not os.path.exists(f"./img/generate/ref_whitebear/no_iterate_xt_next_generate/seed_{seed}"):
        os.makedirs(f"./img/generate/ref_whitebear/no_iterate_xt_next_generate/seed_{seed}")
    x_t_next_images[0].save(f"./img/generate/ref_whitebear/no_iterate_xt_next_generate/seed_{seed}/no_iterate_xt_next_generate_timestep{t}.png")