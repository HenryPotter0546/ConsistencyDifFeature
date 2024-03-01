from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusionPipeline, LCMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import KarrasDiffusionSchedulers
from diffusers import DiffusionPipeline
from diffusers import LatentConsistencyModelPipeline
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LCMScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from src.Myscheduler import Myscheduler
from src.my_unet import MyUNet2DConditionModel
import torch

#pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
# To save GPU memory, torch.float16 can be used, but it may compromise image quality.
#pipe.to(torch_device="cuda", torch_dtype=torch.float32)

#torch_device = "cuda"

class MySDPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae, 
            text_encoder, 
            tokenizer, 
            unet, 
            scheduler, 
            safety_checker, 
            feature_extractor
        )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )       

    @torch.no_grad()
    def lcm_one_step(
        self,
        x0_latent: torch.FloatTensor,
        t,
        up_ft_indices = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        additional_embeds: Optional[Tuple[str, Any]] = None,
        output_type: Optional[str] = "latent",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        device = self._execution_device
        prompt_embeds, _ = additional_embeds

        sampled_noise = torch.randn_like(x0_latent).to(device)
        t = torch.tensor(t, dtype=torch.long, device=device)
        x_t_latent = self.scheduler.add_noise(x0_latent, sampled_noise, t)
        unet_out_dict = self.unet(
            x_t_latent,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict  = False
        )[1]
        model_pred = unet_out_dict["model_pred"]
        unet_layers = unet_out_dict["unet_layers"]
        # for key in unet_layers.keys():
        #     print(f"unet_layer_num{key}.shape", unet_layers[key].shape)
        
        # compute the previous noisy sample x_t -> x_t-1
        # x_t_next_latent, x_0_predicted_latent = self.scheduler.step(model_pred, t, x_t_latent, return_dict=False)
        x_t_next_latent = self.scheduler.step(model_pred, t, x_t_latent, return_dict=False)[0]

        latent = x_t_next_latent

        # Choose use latent or image as output
        if not output_type == "latent":
            latent = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
            # Postprocess image
            image = self.image_processor.postprocess(latent, output_type=output_type)
        else:
            image = latent
            return unet_layers

        return image
    

    @torch.no_grad()
    def lcm_generation_one_step(
        self,
        x_t_latent: torch.FloatTensor,
        t,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        additional_embeds: Optional[Tuple[str, Any]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        device = self._execution_device
        prompt_embeds, _ = additional_embeds

        t = torch.tensor(t, dtype=torch.long, device=device)
        unet_out_dict = self.unet(
            x_t_latent,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict  = False
        )[1]
        model_pred = unet_out_dict["model_pred"]
        unet_layers = unet_out_dict["unet_layers"]
        print("running unet")
        
        # compute the previous noisy sample x_t -> x_t-1
        # x_t_next_latent, x_0_predicted_latent = self.scheduler.step(model_pred, t, x_t_latent, return_dict=False)
        x_t_next_latent = self.scheduler.step(model_pred, t, x_t_latent, return_dict=False)[0]

        x_0_predicted_latent = None
        output ={}
        output["x_t_next_latent"] = x_t_next_latent
        output["x_0_predicted_latent"] = x_0_predicted_latent
        output["model_pred"] = model_pred
        output["unet_layers"] = unet_layers
        
        return output
    
    # @torch.no_grad()
    # def 