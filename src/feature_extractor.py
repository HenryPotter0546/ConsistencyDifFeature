import torch
from PIL import Image
from src.pipe_utils import pipe_selector
from src.new_resnet import set_timestep, init_resnet_func
import torchvision
import tqdm

class CDHFExtractor:
    def __init__(self, config, use_xformers=False):
        self.device="cuda:0"
        self.save_timestep = config["save_timestep"]
        self.extract_mode = config["extract_mode"]
        pipe_dict = pipe_selector(config["lcm_model_name"])
        self.pipe = pipe_dict["pipe"]
        self.use_lora = pipe_dict["use_lora"]
        self.is_sdxl = pipe_dict["is_sdxl"]
        self.lcm_name = pipe_dict["lcm_model_name"]
        self.output_resolution = config["output_resolution"]
        self.num_time_steps = config["num_timesteps"]
        self.idxs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        if use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()
        if self.is_sdxl:
            self.img_size = 768    #512, 768, 1024
        else:
            self.img_size = 512    #512, 768, 1024
        self.pipe.to(torch_device=self.device, torch_dtype=torch.float16)

    def process_image(self, image_pil, res=None, range=(-1, 1)):
        if res:
            image_pil = image_pil.resize(res, Image.BILINEAR)
        image = torchvision.transforms.ToTensor()(image_pil) # range [0, 1]
        r_min, r_max = range[0], range[1]
        image = image * (r_max - r_min) + r_min # range [r_min, r_max]
        return image[None, ...], image_pil
    
    def image2tensor(self, image_path):
        # load image from file
        imgs = []
        image_size = (self.img_size, self.img_size)
        image_pil = Image.open(image_path).convert('RGB')
        img, _ =  self.process_image(image_pil, res=image_size)
        img = img.to(self.device)
        imgs.append(img)
        imgs = torch.vstack(imgs)
        images = torch.nn.functional.interpolate(imgs, size=self.img_size, mode="bilinear")
        image_tensors = images.to(torch.float16)
        return image_tensors
    
    def encode_prompt_general(self, prompt):
        if self.is_sdxl:
            additional_embeds = self.pipe.latent_and_text_preprocess(prompt = prompt, guidance_scale=8.0, use_lora=self.use_lora, height=self.img_size, width=self.img_size)
        else:
            additional_embeds = self.pipe.encode_prompt(prompt=prompt,device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)
        return additional_embeds

    def collect_and_resize_feats(self, unet, idxs, timestep, resolution=-1):
        latent_feats = self.collect_feats(unet, idxs=idxs)
        latent_feats = [feat[timestep] for feat in latent_feats]   # latent_feats[0][49].shape is torch.Size([1, 1280, 8, 8])
        # print(len(latent_feats))
        # for layer_feat in latent_feats:
        #     print(layer_feat.shape)
        if resolution > 0:
            latent_feats = [torch.nn.functional.interpolate(latent_feat, size=resolution, mode="bilinear") for latent_feat in latent_feats]
        # print(len(latent_feats))
        # for layer_feat in latent_feats:
        #     print(layer_feat.shape)
        latent_feats = torch.cat(latent_feats, dim=1)
        # print("final shape", latent_feats.shape)
        return latent_feats

    def collect_feats(self, unet, idxs):
        feats = []
        layers = self.collect_layers(unet, idxs)
        for module in layers:
            feats.append(module.feats)
        return feats

    def collect_layers(self, unet, idxs=None):
        layers = []
        for i, up_block in enumerate(unet.up_blocks):
            for j, module in enumerate(up_block.resnets):
                if idxs is None or (i, j) in idxs:
                    layers.append(module)
        return layers
    
    def ddim_inv_step(self, xt, et, at, at_next, eta):
        """
        Uses the DDIM formulation for sampling xt_next
        Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
        """
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        if eta == 0:
            c1 = 0
        else:
            c1 = (
            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
        return x0_t, xt_next
    
    def consistency_inv_step(self, x0_t, et, at, at_next, eta=0):
        """
        Uses consistency formulation for predicting x0_t and use DDIM inv to do the inversion
        """
        # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        # if eta == 0:
        #     c1 = 0
        # else:
        #     c1 = (
        #     eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        #     )
        c1 = 0 # eta always 0
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
        return x0_t, xt_next
    
    @torch.no_grad()
    def forward(self, 
               images: torch.Tensor,
               prompt: str = '',
    ):
        # 1. encode image and prompt
        images = torch.nn.functional.interpolate(images, size=512, mode="bilinear")
        images = images.to(torch.float16).to("cuda:0")
        img_latent = self.pipe.vae.encode(images).latent_dist.sample(generator=None) * 0.18215
        additional_embeds = self.encode_prompt_general(prompt)

        # 2. prepare scheduler
        self.pipe.scheduler.set_timesteps(self.num_time_steps, self.device)
        timesteps = self.pipe.scheduler.timesteps

        # 3. Unet reconstruction
        self.pipe.output_resolution = self.output_resolution
        init_resnet_func(self.pipe.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        save_timestep_tmp = self.save_timestep
        # 4. select extract_mode and correct unet layers
        # "iid_noise_denoise",   # "iid_noise_denoise", "consistency_generation", "ddim_inversion", "dual_unet_smoothing", "w/o_noise_denoise"
        if self.extract_mode == "iid_noise_denoise":
            timesteps_indices = self.save_timestep
            for i, t_id in enumerate(timesteps_indices):
                t = timesteps[t_id]
                sampled_noise = torch.randn_like(img_latent).to(self.device)
                x_t_latent = self.pipe.scheduler.add_noise(img_latent, sampled_noise, t)
                set_timestep(self.pipe.unet, t_id)
                _ = self.pipe.lcm_generation_one_step(x_t_latent, t, additional_embeds=additional_embeds)
        elif self.extract_mode == "consistency_generation":
            # flip save_timestep list
            timesteps_indices = self.save_timestep[::-1]
            sampled_noise = torch.randn_like(img_latent).to(self.device)
            x_t_latent = self.pipe.scheduler.add_noise(img_latent, sampled_noise, timesteps[timesteps_indices[0]])
            for i, t_id in enumerate(timesteps_indices):
                t = timesteps[t_id]
                set_timestep(self.pipe.unet, t_id)
                output = self.pipe.lcm_generation_one_step(x_t_latent, t, additional_embeds=additional_embeds)
                x_t_latent = output["x_t_next_latent"]
                # x_0_predicted = output["x_0_predicted_latent"]
                # img_latent = x_0_predicted  # update x_0_predicted
        elif self.extract_mode == "ddim_inversion":
            timesteps_indices = self.save_timestep
            # timesteps_indices_next = timesteps_indices + [timesteps_indices[-1] + 1]
            sampled_noise = torch.randn_like(img_latent).to(self.device)
            x_t_latent = self.pipe.scheduler.add_noise(img_latent, sampled_noise, timesteps[timesteps_indices[0]])
            for i, t_id in enumerate(timesteps_indices):
                t = timesteps[t_id]
                t_next = timesteps[t_id-5]
                set_timestep(self.pipe.unet, t_id)
                output = self.pipe.lcm_generation_one_step(x_t_latent, t, additional_embeds=additional_embeds)
                x0_t = output["x_0_predicted_latent"]
                et = output["model_pred"]
                at = self.pipe.scheduler.alphas_cumprod[t]
                at_next = self.pipe.scheduler.alphas_cumprod[t_next]
                x0_t, xt_next = self.consistency_inv_step(x0_t, et, at, at_next, eta=0)
                x_t_latent = xt_next
        elif self.extract_mode == "dual_unet_smoothing_generation":
            # flip save_timestep list
            timesteps_indices = self.save_timestep[::-1]
            sampled_noise = torch.randn_like(img_latent).to(self.device)
            x_t_latent = self.pipe.scheduler.add_noise(img_latent, sampled_noise, timesteps[timesteps_indices[0]])
            for i, t_id in enumerate(timesteps_indices):
                t = timesteps[t_id]

                # generation process at t
                set_timestep(self.pipe.unet, t_id)
                output = self.pipe.lcm_generation_one_step(x_t_latent, t, additional_embeds=additional_embeds)

                # iid noise denoise at t
                sampled_noise = torch.randn_like(img_latent).to(self.device)
                x_t_latent = self.pipe.scheduler.add_noise(img_latent, sampled_noise, t)
                set_timestep(self.pipe.unet, 100+t_id) # set dict for the other unet, add offset to t_id
                _ = self.pipe.lcm_generation_one_step(x_t_latent, t, additional_embeds=additional_embeds)

                # set next xt for next generation process
                x_t_latent = output["x_t_next_latent"]
            # expand save_timestep list
            negative_save_timestep = [(100+x) for x in self.save_timestep]
            save_timestep_tmp = self.save_timestep + negative_save_timestep

        elif self.extract_mode == "dual_unet_smoothing_inversion":
            # flip save_timestep list
            timesteps_indices = self.save_timestep
            sampled_noise = torch.randn_like(img_latent).to(self.device)
            x_t_latent = self.pipe.scheduler.add_noise(img_latent, sampled_noise, timesteps[timesteps_indices[0]])
            for i, t_id in enumerate(timesteps_indices):
                t = timesteps[t_id]
                t_next = timesteps[t_id-5]
                # inversion process at t
                set_timestep(self.pipe.unet, t_id)
                output = self.pipe.lcm_generation_one_step(x_t_latent, t, additional_embeds=additional_embeds)
                x0_t = output["x_0_predicted_latent"]
                et = output["model_pred"]
                at = self.pipe.scheduler.alphas_cumprod[t]
                at_next = self.pipe.scheduler.alphas_cumprod[t_next]
                x0_t, xt_next = self.consistency_inv_step(x0_t, et, at, at_next, eta=0)

                # iid noise denoise at t
                sampled_noise = torch.randn_like(img_latent).to(self.device)
                x_t_latent = self.pipe.scheduler.add_noise(img_latent, sampled_noise, t)
                set_timestep(self.pipe.unet, (100+t_id)) # set dict for the other unet
                _ = self.pipe.lcm_generation_one_step(x_t_latent, t, additional_embeds=additional_embeds)

                # set next xt for next inversion process
                x_t_latent = xt_next
            # expand save_timestep list
            negative_save_timestep = [(100+x) for x in self.save_timestep]
            save_timestep_tmp = self.save_timestep + negative_save_timestep

        elif self.extract_mode == "w/o_noise_denoise":
            timesteps_indices = self.save_timestep
            x_t_latent = img_latent
            for i, t_id in enumerate(timesteps_indices):
                t = timesteps[t_id]
                set_timestep(self.pipe.unet, t_id)
                output = self.pipe.lcm_generation_one_step(x_t_latent, t, additional_embeds=additional_embeds)

        else:
            raise NotImplementedError("there is no extract_mode called {}".format(self.extract_mode))

        feats = []
        for timestep in save_timestep_tmp:
            timestep_feats = self.collect_and_resize_feats(self.pipe.unet, self.idxs, timestep, self.output_resolution)
            feats.append(timestep_feats)

        feats = torch.stack(feats, dim=1)
        self.feats = feats
        print(feats.shape)
        # b, s, c, w, h = feats.shape # [batch_size, num_timesteps, channels, w, h]
        return feats

    # for test
    def extract_features(self, image_path, t_id, prompt = '', ):
        image_tensors = self.image2tensor(image_path)
        img_latent = self.pipe.vae.encode(image_tensors).latent_dist.sample(generator=None) * self.pipe.vae.config.scaling_factor

        if self.is_sdxl:
            additional_embeds = self.pipe.latent_and_text_preprocess(prompt = prompt, guidance_scale=8.0, use_lora=self.use_lora, height=self.img_size, width=self.img_size)
        else:
            additional_embeds = self.pipe.encode_prompt(prompt=prompt,device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)

        num_inference_steps = 50
        self.pipe.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.pipe.scheduler.timesteps
        t = timesteps[t_id]

        set_timestep(self.pipe.unet, t_id)
        unet_layers_t = self.pipe.lcm_one_step(img_latent, t, -1, additional_embeds=additional_embeds, output_type="latent")

        return unet_layers_t