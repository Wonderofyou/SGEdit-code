import torch
from diffusers.models.attention_processor import Attention
from typing import Callable, Optional, Union

class AttnProcessor:
    def __init__(self, controller=None, place_in_unet=None):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        ###### Modification Here ######
        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = attn.batch_to_head_dim(hidden_states)
        assert attention_mask == None
        sim = torch.einsum('b i d, b j d -> b i j', query, key) * attn.scale
        attention_probs = sim.softmax(dim=-1)
        hidden_states = self.controller(query, key, value, sim, attention_probs, is_cross, 
                                        self.place_in_unet, attn.heads, scale=attn.scale)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def register_attention_control(model, controller):
    attn_procs = model.unet.attn_processors
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue
        cross_att_count += 1
        attn_procs[name] = AttnProcessor(
            controller=controller, place_in_unet=place_in_unet
        )
    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count
