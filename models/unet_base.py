import torch
import torch.nn as nn
from model.blocks import DownBlock, MidBlock, UpBlockUnet
from model.blocks import get_time_embedding

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
        
        self.t_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        
        self.up_sample = list(reversed(self.down_sample))
        
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    down_sample=self.down_sample[i],
                    attn=self.attns[i],
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels,
                    time_emb_dim=self.time_emb_dim
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
                    norm_channels=self.norm_channels,
                    num_layers=self.num_mid_layers
                )
            )
        self.ups = nn.ModuleList([])
        
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                self.time_emb_dim, self.num_up_layers,
                self.norm_channels
            )
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size = 3, padding=1)
        
    def forward(self, x, t):
        out = self.conv_in(x)
        
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        
        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out
       