#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:00:07 2019

@author: shariba
"""
import torch
import torch.nn as nn
from deformConv import DeformConv2d
import torch.nn.functional as F

# these are for transformations (can be sent to the model.py)
from warpImagePytorch_bicubic_ST import bicubicImagewarper
from bicubic_interp_2d_t import bicubicInterpolation2D

class warper():
    def __init__(self, device):
        self.device = device
        
    def warpImagePyTorch_bicubic (self, x1, x):
        """V: defomation field """
        out_size = [28, 28]
        V_t= bicubicInterpolation2D.bicubic_interp_2d_t(self.device, (x.permute(0,3,2,1)).to(self.device), out_size)
        transformedImg = bicubicImagewarper.transform_torch_bicubic(self.device, V_t.to(self.device), (x1.unsqueeze(3)).to(self.device), [28, 28])
        return transformedImg

    def warpImagePyTorch_bicubic_dvf (self, x):
    #    from bicubic_interp_2d_t import bicubic_interp_2d_t 
        """V: defomation field """    
        out_size = [7, 7]
        bicubicInterpolation2D(self.device)
        x1 = (x.permute(0,3,2,1)).to(self.device)
        dvf = bicubicInterpolation2D.bicubic_interp_2d_t(self.device, input_=x1, new_size=out_size)
        return dvf
    
class warpImage_pyTorchBicubic(nn.Module):
    def __init__(self, device):
        super(warpImage_pyTorchBicubic, self).__init__()
        self.device = device
        warper(device)
        self.warp = warper.warpImagePyTorch_bicubic
    
    def forward(self, x, y):
        x = self.warp (self, x, y)
        return x
    
class warpImage_pyTorchBicubic_dvf(nn.Module):
    def __init__(self, device):
        super(warpImage_pyTorchBicubic_dvf, self).__init__()
        warper(device)
        self.computedvf = warper.warpImagePyTorch_bicubic_dvf
        self.device = device
    
    def forward(self, x):
        x = self.computedvf (self, x)
        return x   

def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 3, 2, 1, bias=False),
    nn.ELU()
  )   

class DIRNetDeform(nn.Module):
    def __init__(self, logger = None, device=None):
        super(DIRNetDeform, self).__init__()
        self.logger = logger
        self.shape = [28, 28]
        self.device = device
        
        # for affine only
        self.localization = nn.Sequential(nn.Conv2d(2, 8, kernel_size=7), nn.MaxPool2d(2,2), nn.ReLU(True), nn.Conv2d(8, 10, kernel_size=5), nn.MaxPool2d(2,2), nn.ReLU(True))
        self.fc_loc = nn.Sequential(nn.Linear(10*3*3, 32), nn.ReLU(True), nn.Linear(32,3*2))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
        
        #TODO: same padding
        self.conv1 = nn.Conv2d(in_channels=2, out_channels = 64, kernel_size = 3, stride = 1, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum = 0.9, eps=1e-05, affine=True)
        self.elu1 = nn.ELU()
        self.avgpooling1 = nn.AvgPool2d(kernel_size = 2 )
        
        """ added a deformable convolution layer """
        self.conv2 = DeformConv2d(64, 128, 3, padding=1, bias=False, modulation=True)
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum = 0.9, eps=1e-05, affine=True)
        self.elu2 = nn.ELU()
        
#       normal conv layer
        self.conv3 = nn.Conv2d(in_channels=128, out_channels = 128, kernel_size = 3, stride = 1, padding=(1,1))
        self.bn3 = nn.BatchNorm2d(num_features=128, momentum = 0.9, eps=1e-05, affine=True)
        self.elu3 = nn.ELU()
        self.avgpooling2 = nn.AvgPool2d(kernel_size = 2 )
        
        """ added a deformable convolution layer """
        self.conv41 = DeformConv2d(128, 64, 3, padding=1, bias=False, modulation=True)
        self.bn41 = nn.BatchNorm2d(num_features=64, momentum = 0.9, eps=1e-05, affine=True)
        self.elu41 = nn.ELU()
        
        # normal conv layer
        self.conv4 = nn.Conv2d(in_channels=64, out_channels = 2, kernel_size = 3, stride = 1, padding=(1,1))
        self.layerwarp = warpImage_pyTorchBicubic_dvf(self.device)
        self.conv4m = nn.ConvTranspose2d(2, 128, 3, 2, padding=(1,1))
        self.conv3m = nn.ConvTranspose2d(128, 64, 3, 1, padding=0)
        self.elu3m = nn.ELU()
        self.conv2m = nn.ConvTranspose2d(64, 32, 3, 2, padding=(1,1))
        self.elu2m = nn.ELU()
        self.conv1m = nn.ConvTranspose2d(32, 2, 2, 1, padding=(1,1))
        self.elu1m = nn.ELU()
        self.Decoderlayerwarp = warpImage_pyTorchBicubic(self.device)
        
    def log(self, msg):
        if self.logger:
            self.logger.degug(msg)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10*3*3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)
        
        grid = F.affine_grid(theta, x[:,0,:,:].clone().unsqueeze(1).shape)
        x1 = x.clone()
        x1[:,0,:,:] = F.grid_sample(x[:,0,:,:].clone().unsqueeze(1),grid)[:,0,:,:]
        
        return x1
        
    def forward(self, x1):
        self.log(x1.size())
        
        """ perform affine transformation (output: warped src image) """
        x = self.stn(x1)
        
        """ perform deformable (bicubic spline using conv and deformconv) transformation """
        
        x = self.elu1(self.bn1(self.conv1(x1)))
        x = self.avgpooling1(x)
        x = self.elu2(self.bn2(self.conv2(x)))
        x = self.elu3(self.bn3(self.conv3(x)))
        x = self.avgpooling2(x)
        x = self.elu41(self.bn41(self.conv41(x)))
        x = self.conv4(x)
        x = self.layerwarp( x)
        
        useDecode = 1
        if useDecode:
            x = x.permute(0,3,2,1)
            # include decoder layer
            x = self.conv4m(x)
            x = self.conv3m(x)
            x = self.elu3m(x)
            x = self.conv2m(x)
            x = self.elu2m(x)
            x = self.conv1m(x)
            x = self.elu1m(x)
            x = self.Decoderlayerwarp(x1[:,0,:,:], x)
        return x
    
  
class DIRNet(nn.Module):
    def __init__(self, logger = None, device=None):
        super(DIRNet, self).__init__()
        self.logger = logger
        self.shape = [28, 28]
        self.device = device
        
        
        #TODO: same padding
        self.conv1 = nn.Conv2d(in_channels=2, out_channels = 64, kernel_size = 3, stride = 1, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum = 0.9, eps=1e-05, affine=True)
        self.elu1 = nn.ELU()
        self.avgpooling1 = nn.AvgPool2d(kernel_size = 2 )
        
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels = 128, kernel_size = 3, stride = 1, padding=(1,1))
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum = 0.9, eps=1e-05, affine=True)
        self.elu2 = nn.ELU()
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels = 128, kernel_size = 3, stride = 1, padding=(1,1))
        self.bn3 = nn.BatchNorm2d(num_features=128, momentum = 0.9, eps=1e-05, affine=True)
        self.elu3 = nn.ELU()
        self.avgpooling2 = nn.AvgPool2d(kernel_size = 2 )
        self.conv4 = nn.Conv2d(in_channels=128, out_channels = 2, kernel_size = 3, stride = 1, padding=(1,1))
        self.layerwarp = warpImage_pyTorchBicubic_dvf(self.device)
        
        # decoder layers
        self.conv4m = nn.ConvTranspose2d(2, 128, 3, 2, padding=(1,1))
        
        self.conv3m = nn.ConvTranspose2d(128, 64, 3, 1, padding=0)
        self.elu3m = nn.ELU()
        
        self.conv2m = nn.ConvTranspose2d(64, 32, 3, 2, padding=(1,1))
        self.elu2m = nn.ELU()
    
        self.conv1m = nn.ConvTranspose2d(32, 2, 2, 1, padding=(1,1))
        self.elu1m = nn.ELU()
        
        # warp decoded dvf to image
        self.Decoderlayerwarp = warpImage_pyTorchBicubic(self.device)

        
    def log(self, msg):
        if self.logger:
            self.logger.degug(msg)
        
    def forward(self, x1):
        self.log(x1.size())
        x = self.elu1(self.bn1(self.conv1(x1)))
        x = self.avgpooling1(x)
        x = self.elu2(self.bn2(self.conv2(x)))
        x = self.elu3(self.bn3(self.conv3(x)))
        x = self.avgpooling2(x)
        x = self.conv4(x)
        x = self.layerwarp(x.to(self.device))
        
        useDecode = 1
        if useDecode:
            x = x.permute(0,3,2,1)
            
            # include decoder layer
            x = self.conv4m(x)
            
            x = self.conv3m(x)
            x = self.elu3m(x)
        
            x = self.conv2m(x)
            x = self.elu2m(x)

            x = self.conv1m(x)
            x = self.elu1m(x)
            
            x = self.Decoderlayerwarp((x1[:,0,:,:]).to(self.device), x.to(self.device))
        return x
    