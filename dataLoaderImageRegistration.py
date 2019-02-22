#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:38:53 2019

@author: shariba
"""
from data import MNISTDataHandler
import numpy as np

class MnistImageRegistrationDataloader():
    """ Load Mnist pair data for image registration """
    
    def __init__(self, batch_size, iterations):
        self.batch_size = batch_size
        self.iter = iterations
        self.len = self.batch_size * self.iter + self.batch_size
        dh = MNISTDataHandler("../DIRNet-Keras/MNIST_data", is_train=True)
        batch_x, batch_y = dh.sample_pair(self.batch_size)
        self.xy_batch= np.concatenate([batch_x, batch_y], axis = 1)
        for i in range (self.iter):
            batch_x, batch_y = dh.sample_pair(self.batch_size)
            self.xy_batch= np.concatenate([self.xy_batch, np.concatenate([batch_x, batch_y], axis = 1)], axis = 0)
        print("length-->", self.xy_batch.shape[0])   
            
            
    def __getitem__(self, index):
        return self.xy_batch[index]
    
    
    def __len__(self):
        return self.len
    
    
    
if __name__ == "__main__":
    dataset = MnistImageRegistrationDataloader()
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle = True, num_workers=2)
            
    for i, data in enumerate(train_dataloader, 0):
        inputs = data
        
            
