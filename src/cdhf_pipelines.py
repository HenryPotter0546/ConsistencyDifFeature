from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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


class MyLCMPipeline(LatentConsistencyModelPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: LCMScheduler,
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
    def lcm_generation(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 4,
        original_inference_steps: int = None,
        output_type: Optional[str] = "pil",
        latents = None,
        prompt_embeds = None,
        timesteps = None,
        height = None,
        width = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        
        # set batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        dtype = self.dtype
        feature_dim = self.unet.config.sample_size*self.vae_scale_factor        # 768(default) = 96*8 = unet_output_dim * vae_scale_factor
        height_latent = (height or feature_dim) // self.vae_scale_factor        # height/vae_scale_factor or 96
        width_latent = (width or feature_dim) // self.vae_scale_factor          # width/vae_scale_factor or 96
        xT_noises_shape = (batch_size, 4, height_latent, width_latent)          # [batch_size, 4, 96, 96] = (batch, channel, height_latent, width_latent)

        if prompt_embeds is None:
            prompt_embeds, _ = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False)  # [batch_size, 77, 768] = (batch, text_embed_dim, feature_dim)
        
        xT_noises = torch.randn(xT_noises_shape, device=device, dtype=dtype)

        latents = xT_noises
        # print("prompt.shape", prompt_embeds.shape)
        # print("latents.shape", latents.shape)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device)
        timesteps = self.scheduler.timesteps

        # LCM MultiStep Sampling Loop:
        for i, t in enumerate(timesteps):
            unet_out_dict = self.unet(
                latents,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict  = False
            )[1]  # [batch_size, 4, 96, 96] = (batch, channel, unet_output_dim, unet_output_dim)
            
            model_pred = unet_out_dict["model_pred"]
            unet_layers = unet_out_dict["unet_layers"]
            for key in unet_layers.keys():
                print(f"unet_layer_num{key}.shape", unet_layers[key].shape)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents, x_0_predicted = self.scheduler.step(model_pred, t, latents, return_dict=False)

        # Choose use latent or image as output
        if not output_type == "latent":
            image = self.vae.decode(x_0_predicted / self.vae.config.scaling_factor, return_dict=False)[0]
            # Postprocess image
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = x_0_predicted

        return image
    

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
        batch_size = x0_latent.shape[0]
        prompt_embeds, _ = additional_embeds
        # repeat prompt_embeds to batch_size
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)  # [batch_size, 77, 768] = (batch, text_embed_dim, feature_dim)

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
        x_t_next_latent, x_0_predicted_latent = self.scheduler.step(model_pred, t, x_t_latent, return_dict=False)

        latent = x_0_predicted_latent

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
        device = "cuda:0"
        prompt_embeds, _ = additional_embeds
        batch_size = x_t_latent.shape[0]
        # repeat prompt_embeds to batch_size
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)  # [batch_size, 77, 768] = (batch, text_embed_dim, feature_dim)

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

        # compute the previous noisy sample x_t -> x_t-1
        x_t_next_latent, x_0_predicted_latent = self.scheduler.step(model_pred, t, x_t_latent, return_dict=False)

        output ={}
        output["x_t_next_latent"] = x_t_next_latent
        output["x_0_predicted_latent"] = x_0_predicted_latent
        output["model_pred"] = model_pred
        output["unet_layers"] = unet_layers
        
        return output
    
    # @torch.no_grad()
    # def 