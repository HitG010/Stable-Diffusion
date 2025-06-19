from __future__ import absolute_import
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import torchvision
import os
import inspect

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def spatial_average(input, keepdim = True):
    return input.mean(dim=(2, 3), keepdim=keepdim)

class vgg16(nn.Module):
    def __init__(self, requires_grad = False, pretrained = True):
        super(vgg16, self).__init__()
        
        # Load the pretrained VGG16 model
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])  
        for x in range(9, 16):  
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        
        out = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = out(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out
    
    
class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        
        self.register_buffer('shift', torch.tensor([-0.030, -0.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])
        
    def forward(self, x):
        # Normalize the input tensor
        x = (x - self.shift) / self.scale
        return x
    
class SigleConv(nn.Module):
    def __init__(self, in_channels, out_channels = 1, use_dropout = False):
        super(SigleConv, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # Apply the linear layer
        x = self.model(x)
        return x
    
# Learning Perceptual Image Patch Similarity (LPIPS) metric
class LPIPS(nn.Module):
    def __init__(self, net = 'vgg', version = '0.1', use_dropout = True):
        super(LPIPS, self).__init__()
        self.version = version
        # self.net = vgg16(requires_grad=False, pretrained=True)
        self.scaling = ScalingLayer()
        
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.net = vgg16(requires_grad=False, pretrained=True)
        
        # Add 1x1 conv layers
        self.lin0 =  SigleConv(self.chns[0], use_dropout=use_dropout, out_channels=1)
        self.lin1 =  SigleConv(self.chns[1], use_dropout=use_dropout, out_channels=1)
        self.lin2 =  SigleConv(self.chns[2], use_dropout=use_dropout, out_channels=1)
        self.lin3 =  SigleConv(self.chns[3], use_dropout=use_dropout, out_channels=1)
        self.lin4 =  SigleConv(self.chns[4], use_dropout=use_dropout, out_channels=1)

        self.lins = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])
        
        model_path = os.path.abspath(
            os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)    )
        )
        
        print("Loading LPIPS model from: ", model_path)
        
        self.load_state_dict(torch.load(model_path, map_location=device), strict=False) 
        
    def forward(self, x, y, normalize = False):
        
        if normalize:
            x = 2 * x - 1
            y = 2 * y - 1
            
        # Normalize the inputs according to the imagenet normalization
        
        x_input, y_input = self.scaling(x), self.scaling(y)
        
        out_x = self.net.forward(x_input)
        out_y = self.net.forward(y_input)
        
        features_x, features_y, diffs = {}, {}, {}
        
        # Compute L2 distance for each layer
        for i in range(self.L):
            features_x[i] = nn.functional.normalize(out_x[i], dim=1)
            features_y[i] = nn.functional.normalize(out_y[i], dim=1)
            diffs[i] = (features_x[i] - features_y[i]) ** 2
            
        # 1x1 conv 
        res = [spatial_average(self.lins[i](diffs[i]), keepdim=False) for i in range(self.L)]
        
        val = 0

        # Aggregate the results
        for l in range(self.L):
            val += res[l]
            
        return val