"""
Util functions based on Diffuser framework.
"""


import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from diffusers import StableDiffusionPipeline
from utils.basic_utils import enlarge_bbox
from utils.Segment.seg_utils import segment
import torch.nn.functional as F

def seg_mask_from_z0(x0_image, sam_predictor, large_bbox, DEVICE):
    # seg the mask
    masks = segment(sam_predictor, x0_image, large_bbox)
    fg_blend_mask = np.any(masks, axis=0)
    vis_image = None
    fg_blend_mask = torch.from_numpy(fg_blend_mask)[None, None, :, :].float()
    fg_blend_mask = F.interpolate(fg_blend_mask,(64,64),mode='nearest') > 0.5
    fg_blend_mask = fg_blend_mask.to(DEVICE)
    return fg_blend_mask, vis_image
    
class SGPipeline(StableDiffusionPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image


    def unet_forward(self, latents, t, encoder_hidden_states, **kwargs):
        self.editor.set_kwargs(kwargs)
        return self.unet(latents, t, encoder_hidden_states=encoder_hidden_states)

    @staticmethod
    def get_ca_binary_mask(attention_maps, index_to_alter):
        normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        ret = []
        for idx in index_to_alter:
            attention_map = attention_maps[:,:,idx]
            image = attention_map.clone().detach() #.cpu()
            # image = normalize(torch.sigmoid((normalize(image)-0.5)*10))
            image = normalize(image)
            # image = image / image.max()
            # image = image.numpy().astype(np.uint8)
            image = image > 0.5
            ret.append(image)
        ret = torch.stack(ret)
        return ret

    # classifier-free step
    def cf_step(self, latents, noise_pred, t, guidance_scale):
        # classifier-free guidance
        if guidance_scale > 1.:
            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2)
            noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

        # compute the previous noise sample x_t -> x_t-1
        latents, pred_x0 = self.step(noise_pred, t, latents)
        return latents, pred_x0
    
    def attn_modulated_forward(self, latent_model_input, i, t, reg_prompt_embeds_arr, text_embeddings, fg_blend_mask):
        reg_prompt_embeds_fgs, reg_prompt_embeds_bg = reg_prompt_embeds_arr[:-1], reg_prompt_embeds_arr[-1]
        # forward the foreground objects
        if self.editor.use_multi_sampler and i < self.editor.multi_step:
            for rg_i, (rg_embeds, mask) in enumerate(zip(reg_prompt_embeds_fgs, self.editor.layouts)):
                rg_pred = self.unet_forward(latent_model_input, t, encoder_hidden_states=rg_embeds, rg_i=rg_i).sample
                if rg_i == 0:
                    noise_pred = rg_pred
                else:
                    noise_pred = torch.where(mask[None, :, :]==1, rg_pred, noise_pred)
        else:
            noise_pred = self.unet_forward(latent_model_input, t, encoder_hidden_states=text_embeddings, rg_i=-1).sample
        
        # forward then blend with nothing
        bg_pred = self.unet_forward(latent_model_input, t, encoder_hidden_states=reg_prompt_embeds_bg).sample
        noise_pred = torch.where(fg_blend_mask, noise_pred, bg_pred)
        return noise_pred
        
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        editor,
        use_seg=True,
        neg_prompt=None,
        batch_size=1,
        generator=None,
        sam_predictor=None,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        regional_prompts = [],
        bg_latents=None,
        bg_preserve_start=20,
        bg_preserve_end=45,
        **kwds):
        
        DEVICE = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
                
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
                
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
        
        # text embeddings
        reg_prompt_embeds_arr = []
        for rg_prompt in regional_prompts:
            text_input = self.tokenizer(
                rg_prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            rg_embeds = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
            rg_embeds = torch.cat([unconditional_embeddings, rg_embeds])
            reg_prompt_embeds_arr.append(rg_embeds)
        
        # prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                DEVICE,
                generator
            )

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        self.editor = editor    
        
        latents_list = [latents]
        large_bbox = enlarge_bbox(editor.boxes, alpha=1.1)
        fg_blend_mask = torch.any(editor.layouts==1, axis = 0)[None, :, :]
        vis_image = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # Text-to-image forwarding
                if i < editor.attnMod_step: # position control stage
                    noise_pred = self.attn_modulated_forward(latent_model_input, i, t, reg_prompt_embeds_arr, text_embeddings, fg_blend_mask)
                else:
                    noise_pred = self.unet_forward(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                
                latents, pred_x0 = self.cf_step(latents, noise_pred, t, guidance_scale)
                
                # background perservation stage
                if i>=bg_preserve_start and i<bg_preserve_end:
                    if i == bg_preserve_start:
                        x0_image = self.latent2image(pred_x0, return_type="np")
                        if use_seg:
                            fg_blend_mask, vis_image = seg_mask_from_z0(x0_image, sam_predictor, large_bbox, DEVICE)
                            vis_image = Image.fromarray(x0_image)
                        else:
                            vis_image = Image.fromarray(x0_image)
                    latents = torch.where(fg_blend_mask, latents, bg_latents[i+1])
                
                # update the counter
                self.editor.update_step_counter()
                    
                if return_intermediates:
                    latents_list.append(latents)
                progress_bar.update()
            
        image = self.latent2image(latents, return_type="np")
        return image, latents_list, vis_image, fg_blend_mask
 
    
    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = self._execution_device
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
    
    @torch.no_grad()
    def erase(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        blend_step=0,
        latents=None,
        mask_s=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        save_latents=False,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents[1:]]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                if mask_s is not None and i<blend_step: 
                    latents_cur = torch.where((mask_s==1)[None, None, :, :], latents_ref, latents_cur) 
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict tghe noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents[1:])
            # pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if save_latents:
            return image, latents_list
        return image