import torch
import torch.nn as nn
from model.blocks import DownBlock, MidBlock, UpBlock

class VAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        self.attns = model_config['attn_down']
        
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']  
        self.num_heads = model_config['num_heads']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        self.up_sample = list(reversed(self.down_sample))
        
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding = (1, 1))
        
        self.encoder_layers = nn.ModuleList()
        
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(
                DownBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    down_sample=self.down_sample[i],
                    attn=self.attns[i],
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels
                )
            )
        
        self.encoder_mid = nn.ModuleList([])
        
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mid.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i + 1],
                    time_emb_dim = None, 
                    num_heads = self.num_heads, 
                    norm_channels = self.norm_channels,
                    num_layers=self.num_mid_layers
                )
            )
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels * 2, kernel_size=3, padding=1)
        
        self.pre_quant_conv = nn.Conv2d(2 * self.z_channels, 2 * self.z_channels, kernel_size=1)
        
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        self.decoder_mids = nn.ModuleList([])
        
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i - 1],
                    time_emb_dim = None, 
                    num_heads = self.num_heads, 
                    norm_channels = self.norm_channels,
                    num_layers=self.num_mid_layers
                )
            )
            
        self.decoder_layers = nn.ModuleList([])
        
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(
                UpBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i - 1],
                    up_sample=self.down_sample[i - 1],
                    attn=self.attns[i - 1],
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels,
                    num_layers=self.num_up_layers
                )
            )
            
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
        
    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mid:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device = x.device)
        return sample, out
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out
    
    def forward(self, x):
        z, encoder_out = self.encode(x)
        out = self.decode(z)
        return out, encoder_out