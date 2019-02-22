#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 18:05:17 2019

@author: shariba

module load cuda/9.0
source activate TFPytorchGPU

"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from losses import nccLoss_clipVersion, nccLoss
from utils import get_logger

# set your gpu/cpu
from config import get_config
config = get_config(is_train=True)
device = config.device
if device == 'cpu':
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id
    device = torch.device('cuda')
print("Device recognised as:", device)

# parameters here (see config.py for details)
num_epochs = config.epochs
n_batches = config.batch_size_data
iterations = config.iters
n_batches_epoch = config.batch_size_perEpoch
lr = config.lr
num_workers = config.number_workers

# call model
from model import DIRNetDeform
logger = get_logger() 
net = DIRNetDeform(logger, device).to(device)
optimizer = optim.Adam(net.parameters(), lr= lr)

# reload your trained model (take from the argument)
retrain=0
if retrain:
    checkpoint = torch.load('dirConvnet_model-zncc_unet_deformConv_affine-mnist_gpu.ckpt')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

total_loss = []
total_acc = 0

""" Load data """
from dataLoaderImageRegistration import MnistImageRegistrationDataloader
dataset = MnistImageRegistrationDataloader(n_batches, iterations)
# 6000 * 64 pairs
batchedSamples =int((iterations*n_batches)//n_batches_epoch)

print('using traing data loader..., with batch {} and number of worker{} and iterations per epoch will be {}'.format(n_batches_epoch, num_workers, batchedSamples))

train_dataloader = DataLoader(dataset=dataset, batch_size=n_batches_epoch, shuffle = True, num_workers=num_workers)

for epoch in range(num_epochs): 
    total_loss = []
    for i , data in enumerate (train_dataloader):
        inputs = data.to(device)  
        net.train()
        optimizer.zero_grad()
        Iwarped = net(inputs)
        loss = nccLoss_clipVersion(inputs[:,1,:,:].to(device) , Iwarped[:,:,:,0].to(device) )
        loss.backward()
        # update weigths
        optimizer.step()
        total_loss.append(loss.item())
        
        if (epoch + 1) % 5 == 0 and (i + 1) % (batchedSamples//2) == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'dirConvnet_model-zncc_unet_deformConv_affine-mnist_gpu.ckpt')
            net.eval()
            with torch.no_grad():
                Iwarped = net(inputs)
                loss =  nccLoss(inputs[:,1,:,:], Iwarped[:,:,:,0])
                print('evaluation loss is', loss)

        if (i + 1) %  int(batchedSamples//4) == 0:
            print('Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(epoch + 1,num_epochs, i + 1, np.mean(total_loss)))
            
        
