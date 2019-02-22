#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:28:37 2019

@author: shariba
"""

""" Bicubic interpolation and image warping in pytorch 
    FileName: warpImagePytorch_bicubic_ST.py
"""
import torch


""" Bicubic interpolation  of  the vector field coming from convNet is done in this function """
from bicubic_interp_2d_t import bicubicInterpolation2D

    
class bicubicImagewarper():
    def __init__(self, device):
        self.device = device
        bicubicInterpolation2D(self.device)
        
    def repeat_torch(device, x, n_repeats):
        """ repeats n-dim vector with number of batch sizes"""
        dtypeInt = torch.IntTensor
    #    x = x.to('cuda')
        rep_t = torch.t (torch.unsqueeze (torch.ones(n_repeats) ,1))
        rep_t = rep_t.type(dtypeInt)
        x_t = x.type(dtypeInt)
        x_tt = torch.reshape(x_t, [-1, 1] )
        x_t = torch.matmul(x_tt, rep_t)
        return torch.reshape(x_t, [-1])
        
    def interpolate_torch(device,im_t, x_t, y_t, out_size):
        """ interpolation for image warping with the vector field """
        im_t = im_t.to(device)
        x_t = x_t.to(device)
        y_t = y_t.to(device)

        dtypeInt = torch.IntTensor
        if str(device) == 'cuda':
            dtypeFloat = torch.cuda.FloatTensor
            dtypeLong = torch.cuda.LongTensor
        else:
            dtypeFloat = torch.FloatTensor
            dtypeLong = torch.LongTensor
    
        num_batch_t = torch.tensor(im_t.shape[0])
        height_t =  torch.tensor(im_t.shape[1])
        width_t =  torch.tensor(im_t.shape[2])
        channels_t =  torch.tensor(im_t.shape[3])
        #TODO remove this torch.tensor only allow type casting
        height_f_t = height_t.type(dtypeFloat)
        width_f_t = width_t.type(dtypeFloat)
        x_t = x_t.type(dtypeFloat)
        y_t = y_t.type(dtypeFloat)
        out_height = out_size[0]
        out_width = out_size[1]
        #
        zero_t = torch.zeros([]).type(dtypeInt)
        max_y_t = torch.tensor(im_t.shape[1]-1).type(dtypeInt)
        max_x_t = torch.tensor(im_t.shape[2]-1).type(dtypeInt)
        x_t = (x_t + 1.0)* (width_f_t) / 2.0
        y_t = (y_t + 1.0)* (height_f_t) / 2.0
        # do sampling (TODO: Shortening)
        x0_t = torch.floor(x_t).type(dtypeInt)
        x1_t = x0_t + 1
        y0_t = torch.floor(y_t).type(dtypeInt)
        y1_t = y0_t + 1
        x0_t = torch.clamp(x0_t, zero_t, max_x_t)
        x1_t = torch.clamp(x1_t, zero_t, max_x_t)
        y0_t = torch.clamp(y0_t, zero_t, max_y_t)
        y1_t = torch.clamp(y1_t, zero_t, max_y_t)
        
        dim2_t = width_t
        dim1_t = width_t * height_t
        
        base_t = bicubicImagewarper.repeat_torch(device, torch.arange(num_batch_t) *dim1_t, out_height*out_width)
        base_y0_t = base_t + y0_t*dim2_t
        base_y1_t = base_t + y1_t*dim2_t
        idx_a_t = base_y0_t + x0_t
        idx_b_t = base_y1_t + x0_t
        idx_c_t = base_y0_t + x1_t
        idx_d_t = base_y1_t + x1_t
        
        im_flat_t = torch.reshape(im_t, [-1, channels_t]).type(dtypeFloat)
        Ia_t = im_flat_t[idx_a_t.type(dtypeLong), :]
        Ib_t = im_flat_t[idx_b_t.type(dtypeLong), :]
        Ic_t = im_flat_t[idx_c_t.type(dtypeLong), :]
        Id_t = im_flat_t[idx_d_t.type(dtypeLong), :]
        
        # typecasting as below
        x0_f_t = x0_t.type(dtypeFloat)
        x1_f_t = x1_t.type(dtypeFloat)
        y0_f_t = y0_t.type(dtypeFloat)
        y1_f_t = y1_t.type(dtypeFloat)
        
        wa_t = (torch.unsqueeze(((x1_f_t-x_t) * (y1_f_t-y_t)), 1)).to(device)
        wb_t = (torch.unsqueeze(((x1_f_t-x_t) * (y_t-y0_f_t)), 1)).to(device)
        wc_t = (torch.unsqueeze(((x_t-x0_f_t) * (y1_f_t-y_t)), 1)).to(device)
        wd_t = (torch.unsqueeze(((x_t-x0_f_t) * (y_t-y0_f_t)), 1)).to(device)
    
        return (wa_t*Ia_t + wb_t*Ib_t + wc_t*Ic_t+ wd_t*Id_t).to(device)
    
    def meshgrid_torch(device, height, width):
        """ meshgrid formatin using width and height (TODO: replace with pytoch nD implementation???)"""
        X1a =  torch.unsqueeze( torch.linspace(-1.0, 1.0, width) , 1)
        x1 = torch.t(X1a).to(device)
        x_t = torch.matmul(torch.ones(height, 1).to(device), x1)
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),torch.ones(1, width))
        x_t_flat = torch.reshape(x_t, (1, -1))
        y_t_flat = torch.reshape(y_t.to(device), (1, -1))
        return torch.cat((x_t_flat, y_t_flat), 0)
        """ 
        Description: this function needs to be called for warping of the image
        I/P: V: vector field (2D) of shape [batch_size, height_u, width_u, channels=2]
             U: Image file (2D) of shape [batch_size, height_i, width_i, channels = 1/3]: 3 not tested???
             out_size : [height_oImg, width_oImg]
        O/P: transformed image [batch_size, heigh_o, width_o, channel]
        
        (see unitTesting.py for validation code!!!!)
        """  
    def transform_torch_bicubic(device,V, U, out_size):
        """ applies the transformation using bicubic interpolation of the vector field"""
        num_batch = U.shape[0]
    #    height = U.shape[1]
    #    width = U.shape[2]
        num_channels = U.shape[3]    
        out_height = out_size[0]
        out_width = out_size[1]
        
        grid_t = bicubicImagewarper.meshgrid_torch(device, out_height, out_width)
        grid_t = torch.reshape(grid_t, [-1,])
        grid_t = torch.unsqueeze(grid_t, 0)
        grid_t = grid_t.repeat(num_batch, 1)
    #    if num_batch >1:
    #        grid_t = torch.stack((grid_t, grid_t), dim=0).view(num_batch, grid_t.shape[0])
            
    
        grid_t = torch.reshape(grid_t, [num_batch, 2, -1])
        # this is displacement vector field
        V_t= bicubicInterpolation2D.bicubic_interp_2d_t(device, V, out_size)
        V_t = V_t.permute(0,3,1,2)
        # reshaping
        V_t = torch.reshape(V_t, [num_batch, 2, -1])
        T_g_t = torch.add(V_t.to(device), grid_t.to(device))
        x_s_t = T_g_t[:,0,:]
        y_s_t = T_g_t[:,1,:]
        x_s_t_flat = torch.reshape(x_s_t, [-1])
        y_s_t_flat = torch.reshape(y_s_t, [-1])
        input_transformed_t = bicubicImagewarper.interpolate_torch(device, U, x_s_t_flat, y_s_t_flat, out_size)
        return torch.reshape(input_transformed_t, [num_batch,out_height,out_width, num_channels ])
