from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, DDPMScheduler
from src.cdhf_pipelines import MyLCMPipeline
from src.my_sd_pipeline import MySDPipeline
from src.my_unet import MyUNet2DConditionModel
from src.myddpm_scheduler import MyDDPMScheduler
import torch

def pipe_selector(lcm_model_name, device="cuda"):
    LCM_MODELS = [  "LCM_Dreamshaper_v7", 
                    "lcm-lora-sdv1-5", 
                    "lcm-sdxl", 
                    "lcm-ssd-1b", 
                    "lcm-lora-sdxl", 
                    "lcm-lora-ssd-1b",
                    "stable-diffusion-v1-5",
                    ]
    assert lcm_model_name in LCM_MODELS, f"---- '{lcm_model_name}' is not in available LCM model list ----"

    use_lcm = True
    if lcm_model_name == "LCM_Dreamshaper_v7":
        #1. Dreamshaper_v7 Full tunned model
        unet = MyUNet2DConditionModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="unet")
        pipe = MyLCMPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", unet=unet)
        use_lora = False
        is_sdxl = False
    elif lcm_model_name == "lcm-lora-sdv1-5":
        # # 2. Dreamshaper_v7_base SD lora model
        # model_id = "Lykon/dreamshaper-7"
        model_id = "runwayml/stable-diffusion-v1-5"
        lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
        # unet = MyUNet2DConditionModel.from_pretrained("Lykon/dreamshaper-7", subfolder="unet")
        # pipe = MyLCMPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16, variant="fp16")
        # pipe.load_lora_weights(lcm_lora_id)
        unet = MyUNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        pipe = MyLCMPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
        # pipe.unet = unet
        pipe.load_lora_weights(lcm_lora_id)
        pipe.fuse_lora(
                fuse_unet=True,
                fuse_text_encoder=True,
                lora_scale=1.0,
                safe_fusing=False,
            )
        use_lora = True
        is_sdxl = False
    elif lcm_model_name == "stable-diffusion-v1-5":
        # 3. normal stable diffusion model
        model_id = "runwayml/stable-diffusion-v1-5"
        unet = MyUNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        pipe = MyLCMPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet)
        # pipe = MySDPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", unet=unet)
        use_lora = False
        is_sdxl = False
        use_lcm = False
    else:
        if lcm_model_name == "lcm-sdxl":
            # 1. SDXL Full tunned model
            unet_id = "latent-consistency/lcm-sdxl"
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            use_lora = False
            is_sdxl = True
        elif lcm_model_name == "lcm-ssd-1b":
            # 2. SDXL_base_SSD-1B Full tunned model
            unet_id = "latent-consistency/lcm-ssd-1b"
            model_id = "segmind/SSD-1B"
            use_lora = False
            is_sdxl = True
        elif lcm_model_name == "lcm-lora-sdxl":
            # 1. SDXL lora model
            unet_id = "latent-consistency/lcm-sdxl"
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
            use_lora = True
            is_sdxl = True
        elif lcm_model_name == "lcm-lora-ssd-1b":
            # 2. SDXL_base_SSD-1B lora model
            unet_id = "latent-consistency/lcm-ssd-1b"
            model_id = "segmind/SSD-1B"
            lcm_lora_id = "latent-consistency/lcm-lora-ssd-1b"
            use_lora = True
            is_sdxl = True

    if use_lcm:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = MyDDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        # pass
        # pipe.scheduler = DDIMScheduler.from_config(
        # pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        # )

    return {"pipe": pipe, "use_lora": use_lora, "lcm_model_name": lcm_model_name, "is_sdxl": is_sdxl}