#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:38:23 2019

@author: shariba
"""
import numpy as np
import torch


""""
---> converting to numpy array has to be done with x.data.numpy() which detaches the gradient path
"""
class bicubicInterpolation2D():
    def __init__(self, device):
        self.device = device
        
    def bicubic_interp_2d_t(device, input_, new_size=[7, 7]):
      """
      Args :
        input_ : Input tensor. Its shape should be
            [batch_size, height, width, channel].
            In this implementation, the shape should be fixed for speed.
        new_size : The output size [new_height, new_width]
      ref : http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
      """
      shape = input_.shape
      batch_size = shape[0]
      height  = shape[1]
      width   = shape[2]
      channel = shape[3]
      
      def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)
      
      def gather(input_s, index):
        idx1, idx2, idx3, idx4 = index.chunk(4, dim=4)
        input_s_gather=input_s[idx1.type(torch.LongTensor), idx2.type(torch.LongTensor), idx3.type(torch.LongTensor), idx4.type(torch.LongTensor)].squeeze(4)
        return input_s_gather
    
    
      def _hermite(device, A, B, C, D, t):
        a = A * -0.5 + B * 1.5 + C * -1.5 + D * 0.5
        b = A + B * -2.5 + C * 2.0 + D * -0.5
        c = A * -0.5 + C * 0.5
        d = B
        a = a.to(device)
        t = t.to(device)
        b = b.to(device)
        c = c.to(device)
        d = d.to(device)
            
        val = a*t*t*t + b*t*t + c*t + d
        return val
    
      def _get_grid_array(n_i, y_i, x_i, c_i):
        # ???????  Todo: convert to pytorch  
        n, y, x, c = np.meshgrid(n_i, y_i, x_i, c_i, indexing='ij')
        n = (np.expand_dims(n, axis=4))
        y = (np.expand_dims(y, axis=4))
        x = (np.expand_dims(x, axis=4))
        c = (np.expand_dims(c, axis=4))
        d = torch.from_numpy(np.concatenate([n,y,x,c], axis=4))
        return d
    
      def _get_frac_array(x_d, y_d, n, c):
          
        x = x_d.shape[0]
        y = y_d.shape[0]
        x_t = torch.reshape(x_d, [1, -1, 1])
        y_t = torch.reshape(y_d, [-1, 1, 1])
        x_t = tile(x_t,0,x)
        x_t = x_t.unsqueeze(0).repeat(n, 1, 1, 1)
        y_t = tile(y_t,1,y)
        y_t = y_t.unsqueeze(0).repeat(n, 1, 1, 1)
        
    #    y_t = torch.from_numpy(np.tile(y_t, (n,1,x,c)))
    #    x_t = torch.from_numpy(np.tile(x_t, (n,y,1,c)))
        return x_t, y_t
    
      def _get_index_tensor(grid, x, y):
        """completely in torch"""
        new_grid = grid
        grid_y = grid[:,:,:,:,1] + y
        grid_x = grid[:,:,:,:,2] + x
        grid_y = torch.clamp(grid_y, 0, height-1)
        grid_x = torch.clamp(grid_x, 0, width-1)
    
        new_grid[:,:,:,:,1] = grid_y
        new_grid[:,:,:,:,2] = grid_x
    
        return new_grid.type(torch.IntTensor)
    
    
      new_height = new_size[0]
      new_width  = new_size[1]
    
    # converted to torch but nD grid by torch (TODO-->)
      n_i = torch.arange(batch_size)
      c_i = torch.arange(channel)
      y_f = torch.linspace(0., height-1, new_height)
      y_i = y_f.type(torch.IntTensor) 
      y_d_t = y_f - torch.floor(y_f)
      x_f = torch.linspace(0., width-1, new_width)
      x_i = x_f.type(torch.IntTensor)
      x_d_t = x_f - torch.floor(x_f)
    
      n_i = np.arange(batch_size)
      c_i = np.arange(channel)
      y_f = np.linspace(0., height-1, new_height)
      y_i = y_f.astype(np.int32)
      x_f = np.linspace(0., width-1, new_width)
      x_i = x_f.astype(np.int32)
    
      grid = _get_grid_array(n_i, y_i, x_i, c_i)   
      x_t, y_t = _get_frac_array(x_d_t, y_d_t, batch_size, channel)
    
      i_00 = _get_index_tensor(grid, -1, -1)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      
      i_10 = _get_index_tensor(grid, +0, -1)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_20 = _get_index_tensor(grid, +1, -1)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_30 = _get_index_tensor(grid, +2, -1)
      
      grid = _get_grid_array(n_i, y_i, x_i, c_i)    
      i_01 = _get_index_tensor(grid, -1, +0)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_11 = _get_index_tensor(grid, +0, +0)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_21 = _get_index_tensor(grid, +1, +0)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_31 = _get_index_tensor(grid, +2, +0)
          
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_02 = _get_index_tensor(grid, -1, +1)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_12 = _get_index_tensor(grid, +0, +1)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_22 = _get_index_tensor(grid, +1, +1)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_32 = _get_index_tensor(grid, +2, +1)
         
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_03 = _get_index_tensor(grid, -1, +2)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_13 = _get_index_tensor(grid, +0, +2)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_23 = _get_index_tensor(grid, +1, +2)
      grid = _get_grid_array(n_i, y_i, x_i, c_i)
      i_33 = _get_index_tensor(grid, +2, +2)
      
    
      p_00 = gather(input_, i_00)
      p_10 = gather(input_, i_10)
      p_20 = gather(input_, i_20)
      p_30 = gather(input_, i_30)
    
      p_01 = gather(input_, i_01)
      p_11 = gather(input_, i_11)
      p_21 = gather(input_, i_21)
      p_31 = gather(input_, i_31)
    
      p_02 = gather(input_, i_02)
      p_12 = gather(input_, i_12)
      p_22 = gather(input_, i_22)
      p_32 = gather(input_, i_32)
    
      p_03 = gather(input_, i_03)
      p_13 = gather(input_, i_13)
      p_23 = gather(input_, i_23)
      p_33 = gather(input_, i_33)
    
      col0 = _hermite(device, p_00, p_10, p_20, p_30, x_t)
      col1 = _hermite(device,p_01, p_11, p_21, p_31, x_t)
      col2 = _hermite(device,p_02, p_12, p_22, p_32, x_t)
      col3 = _hermite(device,p_03, p_13, p_23, p_33, x_t)
      value = _hermite(device, col0, col1, col2, col3, y_t)
      
      return value