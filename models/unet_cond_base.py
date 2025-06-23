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