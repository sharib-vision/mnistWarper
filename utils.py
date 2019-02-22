#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:48:52 2019

@author: shariba
"""
import logging
import imp
import torchvision.transforms as transforms

# redunant function
#https://docs.python.org/3/howto/logging-cookbook.html
def get_logger(ch_log_level=logging.ERROR,
               fh_log_level=logging.INFO):
    logging.shutdown()
    imp.reload(logging)
    logger = logging.getLogger("cheatsheet")
    logger.setLevel(logging.DEBUG)
    

# TODO: work on augmentation/scaling and cropping mostly
def augmentation():    
# see here: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform
    
# TODO: random cropping for large images to compute dvf
#transformed_dataset = dataset(transform=transforms.Compose([Rescale(16),RandomCrop(16),ToTensor()]))