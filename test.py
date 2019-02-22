#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 18:49:16 2019

@author: shariba
"""

from data import MNISTDataHandler, FashionMNISTDataHandler
import numpy as np
import torch
import os

# set your gpu/cpu
def testViz (y_pred, inputImagePair):
    im_size= [inputImagePair[0].shape[1], inputImagePair[0].shape[1]]
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    if y_pred.shape[3] == 2:
        u=np.reshape(y_pred[0][:,:,0], im_size)
        #        img_source = WarpST(inputImagePair, y_pred, im_size)
        plt.imshow(u)
        v=np.reshape(y_pred[0][:,:,1], im_size)
        fig.add_subplot(2, 1, 2)
        plt.imshow(v)
    else:
        img_warped=np.reshape(y_pred[0][:,:,0], im_size)
        plt.title('warp_0->1')
        plt.imshow(img_warped)
    
    #plt original
    fig.add_subplot(2, 2, 1)
    img_orig_sr=np.reshape(inputImagePair[0][:,:,0], im_size)
    plt.imshow(img_orig_sr)
    plt.title('src')
    fig.add_subplot(2, 2, 2)
    img_orig_tr=np.reshape(inputImagePair[0][:,:,1], im_size)
    plt.imshow(img_orig_tr)
    plt.title('tr')
    
    
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from model import DIRNetDeform
    from utils import get_logger
    logger = get_logger() 
    
    model = DIRNetDeform(logger, device='cpu')
    resultDir = 'imgs'
    os.makedirs(resultDir, exist_ok =True)
    
    # set your checkpoint here
    checkpointFileName = 'dirConvnet_model-zncc_SSD_unet-2_gpu_deformConv_affine.ckpt'
#    checkpointFileName = 'dirConvnet_model-zncc_unet_deformConv_affine-mnist_gpu.ckpt'
    
    
    #
    checkpoint = torch.load(checkpointFileName, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(checkpoint['model_state_dict'])
    shape=[28, 28]
    dh = MNISTDataHandler("../DIRNet-Keras/MNIST_data", is_train=False)
    
#    dh = FashionMNISTDataHandler("../fashionMnist", is_train=False)
    
    Iwarped_mean = 0
    targetImage_mean = 0
    sourceImage_mean = 0
    length = 25
    
    for i in range (length):
        batch_x, batch_y = dh.sample_pair(1)
        input2model= torch.FloatTensor(np.concatenate([batch_x, batch_y], axis=1))  
        model.eval()
        
        Iwarped=model(input2model)
        Iwarped = Iwarped.detach().numpy()
        y_pred = np.reshape(Iwarped, shape)
        
        fig = plt.figure()
        fig.add_subplot(3, 3, 1)
        plt.imshow(np.reshape(batch_x, shape), cmap=plt.cm.bone)
        plt.title('src')
        fig.add_subplot(3, 3, 2)
        plt.imshow(np.reshape(y_pred, shape), cmap=plt.cm.bone)
        plt.title('pred')
        fig.add_subplot(3, 3, 3)
        plt.imshow(np.reshape(batch_y, shape), cmap=plt.cm.bone)
        plt.title('tr')
        
        # print the mean shapes here
        Iwarped_mean = Iwarped + Iwarped_mean
        targetImage_mean = batch_y  + targetImage_mean
    
        if i == length-1:
            fig = plt.figure()
            fig.add_subplot(2, 2, 1)
            plt.imshow(np.reshape( Iwarped_mean/(length), [28,28]), cmap=plt.cm.bone) 
            plt.title('warped')
            fig.add_subplot(2, 2, 2)
            plt.imshow(np.reshape( targetImage_mean/(length), [28,28]), cmap=plt.cm.bone) 
            plt.title('target')
            
            meanImage = np.concatenate((255*np.reshape(Iwarped_mean/(length), [28,28]), 255*np.reshape( targetImage_mean/(length), [28,28])), axis=1)
            cv2.imwrite(resultDir+'/mean_sample'+'.png', meanImage)
     
        
        vis = np.concatenate((255*np.reshape(batch_x, shape), 255*np.reshape(y_pred, shape)), axis=1)
        vis1 = np.concatenate((vis, 255*np.reshape(batch_y, shape)), axis=1)
        cv2.imwrite(resultDir+'/sample'+str(i)+'.png', vis1)
        
        

        