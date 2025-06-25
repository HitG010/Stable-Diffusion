import torch
import torch.nn as nn
from einops import einsum
from model.blocks import DownBlock, MidBlock, UpBlockUnet
from model.blocks import get_time_embedding
from utils.config_utils import *

class Unet(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.time_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
 
        self.class_cond = False
        self.text_cond = False
        self.image_cond = False
        self.text_embed_dim = None
        self.condition_config = get_config_value(model_config, 'condition_config', None)
        
        
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            
            if 'class' in condition_types:
                validate_class_config(self.condition_config)
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']

            if 'text' in condition_types:
                validate_text_config(self.condition_config)
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
                
            if 'image' in condition_types:
                self.image_cond = True
                self.im_cond_input_channels = self.condition_config['image_condition_config']['image_condition_input_channels']
                self.im_cond_output_channels = self.condition_config['image_condition_config']['image_condition_output_channels']
                
        if self.class_cond:
            self.class_embed = nn.Embedding(self.num_classes, self.time_emb_dim)
            
        if self.image_cond:
            self.cond_conv_in = nn.Conv2d(in_channels = self.im_cond_input_channels,
                                          out_channels = self.im_cond_output_channels,
                                          kernel_size = 1, bias = False)
            self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_channels, 
                                              out_channels = self.down_channels[0], 
                                              kernel_size = 3, padding = 1)
            
        else:
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.cond = self.text_cond or self.class_cond or self.image_cond
        
        
        self.t_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.downs = nn.ModuleList()
        
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    time_emb_dim=self.time_emb_dim,
                    down_sample=self.down_sample[i],
                    attn=self.attns[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                    norm_channels=self.norm_channels,
                    cross_attn=self.text_cond,
                    context_dim=self.text_embed_dim
                )
            )
        
        self.mids = nn.ModuleList([])

        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i + 1],
                    time_emb_dim=self.time_emb_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                    cross_attn=self.text_cond,
                    context_dim=self.text_embed_dim
                )
            )
            
        self.ups = nn.ModuleList([])
        
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlockUnet(
                    in_channels=self.down_channels[i] * 2,
                    out_channels=self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                    up_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_up_layers,
                    norm_channels=self.norm_channels,
                    cross_attn=self.text_cond,
                    context_dim=self.text_embed_dim
                )
            )
            
        self.norm_out = nn.GroupNorm(num_groups=self.norm_channels, num_channels=self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, cond_input = None):
        if self.cond:
            assert cond_input is not None, 'Condition input must be provided when model is conditioned'

        if self.image_cond:
            validate_image_conditional_input(cond_input, x)
            im_cond = cond_input['image']
            im_cond = nn.functional.interpolate(im_cond, size=x.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)
            
            assert im_cond.shape[-2:] == x.shape[-2:], 'Image condition input must match the spatial dimensions of the input image'
            
            x = torch.cat([x, im_cond], dim=1)
            out = self.conv_in_concat(x)
        else:
            out = self.conv_in(x)

        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.time_emb_dim)
        t_emb = self.t_proj(t_emb)
        
    
        # Class condition
        
        if self.class_cond:
            validate_class_conditional_input(cond_input, x, self.num_classes)
            class_embed = einsum(cond_input['class'].float(), self.class_emb.weight, 'b n, n d -> b d')
            t_emb += class_embed
            
        context_hidden_states = None
        if self.text_cond:
            assert 'text' in cond_input, 'Text condition input must be provided when model is text conditioned'
            
            context_hidden_states = cond_input['text']
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb, context_hidden_states=context_hidden_states)
            
        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states=context_hidden_states)
            
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states=context_hidden_states)
            
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        
        return out