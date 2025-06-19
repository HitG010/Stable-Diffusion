import torch
import torch.nn as nn

def get_time_embedding(time_steps, t_emb_dim):
    """
    Generates time embeddings for the given time steps.
    """
    
    assert t_emb_dim % 2 == 0, "t_emb_dim must be even"
    
    factor = 10000 ** (torch.arange(
        start = 0, end = t_emb_dim / 2, device = time_steps.device) / (t_emb_dim // 2))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels, cross_attn = False, context_dim = None):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.time_emb_dim = time_emb_dim
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                )
                for i in range(num_layers)
            ]
        )
        
        if self.time_emb_dim is not None:
            self.time_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.time_emb_dim, out_channels),
                    )
                    for _ in range(num_layers)
                ]
            )
            
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            
        if self.cross_attn:
            assert context_dim is not None, "context_dim must be provided for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            self.context_proj = nn.ModuleList(
                [
                    nn.Linear(context_dim, out_channels)
                    for _ in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()
        
    def forward(self, x, time_emb=None, context=None):
        out = x
        
        for i in range(self.num_layers):
            resnet_in = out
            out = self.resnet_conv_first[i](out)
            if self.time_emb_dim is not None:
                out = out + self.time_emb_layers[i](time_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_in)
            
        if self.attn:
            # attention block of unet
            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn
        
        if self.cross_attn:
            assert context is not None, "context cannot be None if attention layers are used"
            batch_size, channels, height, width = out.shape
            in_cross_attn = out.reshape(batch_size, channels, height * width)
            in_cross_attn = self.cross_attention_norms[i](in_cross_attn)
            in_cross_attn = in_cross_attn.transpose(1, 2)
            assert context.shape[0] == batch_size, "context batch size must match input batch size"
            assert context.shape[1] == self.context_dim, "context dimension must match context_dim"
            context_proj = self.context_proj[i](context)
            out_attn, _ = self.cross_attentions[i](in_cross_attn, context_proj, context_proj)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn
            
        # Downsample the output
        out = self.down_sample_conv(out)
        return out
    
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, 
                 num_heads, num_layers, norm_channels, cross_attn=None, context_dim=None):
        
        super().__init__()
        self.num_layers = num_layers
        self.time_emb_dim = time_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                )
                for i in range(num_layers + 1)
            ]
        )
        
        if self.time_emb_dim is not None:
            self.time_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.time_emb_dim, out_channels),
                    )
                    for _ in range(num_layers + 1)
                ]
            )
            
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ]
        )
            
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        
        if self.cross_attn:
            assert context_dim is not None, "context_dim must be provided for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            self.context_proj = nn.ModuleList(
                [
                    nn.Linear(context_dim, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
        
    def forward(self, x, time_emb=None, context=None):
        
        out = x
        resnet_in = out
        out = self.resnet_conv_first[0](out)
        
        if self.time_emb_dim is not None:
            out = out + self.time_emb_layers[0](time_emb)[:, :, None, None]
        
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_in)
        
        for i in range(self.num_layers):
            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn
            
            if self.cross_attn:
                assert context is not None, "context cannot be None if attention layers are used"
                batch_size, channels, height, width = out.shape
                in_cross_attn = out.reshape(batch_size, channels, height * width)
                in_cross_attn = self.cross_attention_norms[i](in_cross_attn)
                in_cross_attn = in_cross_attn.transpose(1, 2)
                assert context.shape[0] == batch_size, "context batch size must match input batch size"
                assert context.shape[1] == self.context_dim, "context dimension must match context_dim"
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_cross_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
                out = out + out_attn
                
            resnet_in = out
            out = self.resnet_conv_first[i + 1](out)
            if self.time_emb_dim is not None:
                out = out + self.time_emb_layers[i + 1](time_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_in)
        
        return out
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.attn = attn
        self.time_emb_dim = time_emb_dim
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                )
                for i in range(num_layers)
            ]
        )
        
        if self.time_emb_dim is not None:
            self.time_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.time_emb_dim, out_channels),
                    )
                    for _ in range(num_layers)
                ]
            )
            
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        
        self.up_sample_conv = nn.ConvTranspose2d(
            in_channels, in_channels, 4, 2, 1) if up_sample else nn.Identity()
        
    def forward(self, x, out_down = None, time_emb = None):
        
        x = self.up_sample_conv(x)
        
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
            
        out = x
        
        for i in range(self.num_layers):
            resnet_in = out
            out = self.resnet_conv_first[i](out)
            if self.time_emb_dim is not None:
                out = out + self.time_emb_layers[i](time_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_in)
            
            if self.attn:
                # attention block of unet
                batch_size, channels, height, width = out.shape
                in_attn = out.reshape(batch_size, channels, height * width)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
                out = out + out_attn
                
        return out
    
class UpBlockUnet(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_emb_dim,
                 up_sample, num_heads, num_layers, norm_channels, cross_attn=False, context_dim=None):
        
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.time_emb_dim = time_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                )
                for i in range(num_layers)
            ]
        )
        
        if self.time_emb_dim is not None:
            self.time_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.time_emb_dim, out_channels),
                    )
                    for _ in range(num_layers)
                ]
            )
            
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                )
                for _ in range(num_layers)
            ]
        )
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ]
        )
        
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        
        if self.cross_attn:
            assert context_dim is not None, "context_dim must be provided for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            self.context_proj = nn.ModuleList(
                [
                    nn.Linear(context_dim, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        
        self.up_sample_conv = nn.ConvTranspose2d(
            in_channels, in_channels, 4, 2, 1) if up_sample else nn.Identity()
        
    def forward(self, x, out_down=None, time_emb=None, context=None):
        
        x = self.up_sample_conv(x)
        
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
            
        out = x
        
        for i in range(self.num_layers):
            resnet_in = out
            out = self.resnet_conv_first[i](out)
            if self.time_emb_dim is not None:
                out = out + self.time_emb_layers[i](time_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_in)
                # attention block of unet
            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn
                
            if self.cross_attn:
                assert context is not None, "context cannot be None if attention layers are used"
                batch_size, channels, height, width = out.shape
                in_cross_attn = out.reshape(batch_size, channels, height * width)
                in_cross_attn = self.cross_attention_norms[i](in_cross_attn)
                in_cross_attn = in_cross_attn.transpose(1, 2)
                assert len(context.shape) == 3, "context must be a 3D tensor"
                assert context.shape[0] == batch_size, "context batch size must match input batch size"
                assert context.shape[-1] == self.context_dim, "context dimension must match context_dim"
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_cross_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
                out = out + out_attn
                
        return out