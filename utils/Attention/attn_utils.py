import torch.nn.functional as F
import numpy as np
import torch

from einops import rearrange
from PIL import Image

'''
This is modified from:
1. MutualSelfAttention: https://ljzycmd.github.io/projects/MasaCtrl/
2. DenseDiffusion: https://github.com/naver-ai/DenseDiffusion
'''

class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            # self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
    
    def update_step_counter(self):
        self.cur_step += 1
    
    def set_index_to_alter(self, arr):
        pass
    
    def set_kwargs(self, kwargs):
        self.kwargs = kwargs


class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        # print("MasaCtrl at denoising steps: ", self.step_idx)
        # print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)

        return out


class MutualSelfAttentionControlMask(MutualSelfAttentionControl):
    def __init__(self,  start_step=0, start_layer=0, layer_idx=None, step_idx=None, total_steps=50,  mask_s=None, mask_t=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        self.mask_s = mask_s  # source mask with shape (h, w)
        self.mask_t = mask_t  # target mask with same shape as source mask

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten()
            # background
            sim = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out
        
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)
        out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out


class DenseMultiDiff(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self, supr_idxs, layouts, boxes, weight_dtype, use_multi_sampler,
                 bsz=1, sreg=0.3, creg=1., treg=1., multi_step=10, attnMod_step=15, device="cuda"):
        super().__init__()
        self.supr_idxs = supr_idxs
        self.weight_dtype = weight_dtype
        self.device = device
        self.bsz = bsz
        self.layouts = layouts
        self.boxes = boxes
        self.sreg = sreg
        self.creg = creg
        self.treg = treg
        self.multi_step = multi_step
        self.attnMod_step = attnMod_step
        self.use_multi_sampler = use_multi_sampler
        self.sreg_maps, self.reg_sizes, self.creg_maps = self._get_modulation_maps(use_multi_sampler)
        self.cross_attention_store = {}
        
    def _get_modulation_maps(self, use_multi_sampler):
        sreg_maps_list, creg_maps_list, reg_sizes_list = [], [], []
        if use_multi_sampler:
            for i in range(len(self.layouts)):
                sreg_maps, reg_sizes = self.get_SA_suppress(self.layouts[i:i+1])
                creg_maps = self.get_CA_suppress(self.layouts[i:i+1], self.supr_idxs[i])
                sreg_maps_list.append(sreg_maps)
                reg_sizes_list.append(reg_sizes)
                creg_maps_list.append(creg_maps)
                
        sreg_maps, reg_sizes = self.get_SA_suppress(self.layouts)
        creg_maps = self.get_CA_suppress(self.layouts, self.supr_idxs[-1])
        sreg_maps_list.append(sreg_maps)
        reg_sizes_list.append(reg_sizes)
        creg_maps_list.append(creg_maps)
        
        return sreg_maps_list, reg_sizes_list, creg_maps_list
    
    def get_SA_suppress(self, layouts, sp_sz=64):
        sreg_maps = {}
        reg_sizes = {}
        for r in range(4):
            res = int(sp_sz/np.power(2,r))
            layouts_s = F.interpolate(layouts,(res, res),mode='nearest')
            layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(self.bsz,1,1)
            reg_sizes[np.power(res, 2)] = 1-1.*layouts_s.sum(-1, keepdim=True)/(np.power(res, 2))
            sreg_maps[np.power(res, 2)] = layouts_s
        return sreg_maps, reg_sizes
    
    def get_CA_suppress(self, layouts, supr_idxs, sp_sz=64):
        dtype = self.weight_dtype
        pww_maps = torch.zeros((1, 77, sp_sz,sp_sz)).to(self.device)
        for i, idxs in enumerate(supr_idxs):
            pww_maps[:,idxs,:,:] = layouts[i:i+1]
        
        creg_maps = {}
        for r in range(4):
            res = int(sp_sz/np.power(2,r))
            layout_c = F.interpolate(pww_maps,(res,res),mode='nearest').view(1,77,-1).permute(0,2,1).repeat(self.bsz,1,1)
            creg_maps[np.power(res, 2)] = layout_c.to(dtype)
        return creg_maps
    
    def attn_batch(self, i, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        creg = self.creg
        sreg = self.sreg
        treg = self.treg
        creg_maps = self.creg_maps[i]
        reg_sizes = self.reg_sizes[i]
        sreg_maps = self.sreg_maps[i]
        if is_cross:
            min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
            mask = creg_maps[sim.size(1)].repeat(num_heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat(num_heads,1,1)
            
            sim[int(sim.size(0)/2):] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/2):])
            sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/2):]-min_value)
        else:
            min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
            mask = sreg_maps[sim.size(1)].repeat(num_heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat(num_heads,1,1)
            
            sim[int(sim.size(0)/2):] += (mask>0)*size_reg*sreg*treg*(max_value-sim[int(sim.size(0)/2):])
            sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*sreg*treg*(sim[int(sim.size(0)/2):]-min_value)
        
        attn = sim.softmax(-1)
        
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out
    
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):  
        if self.kwargs.get("rg_i") != None:
            rg_i = self.kwargs.get("rg_i")
            self.treg = pow(1 - self.cur_step/50, 5)
            return self.attn_batch(rg_i, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
    
    def aggregate_attention(self, attention_res=16):
        average_attn = []
        for _, attn in self.cross_attention_store.items():
            if attn.shape[1] == attention_res**2:
                average_attn.append(attn)
        average_attn = torch.cat(average_attn, dim=0)
        average_attn = average_attn.sum(0) / average_attn.shape[0]
        average_attn = average_attn.reshape(attention_res,attention_res,77)
        return average_attn
    
