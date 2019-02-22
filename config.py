#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Sun Feb  3 18:05:17 2019
    
    @author: shariba
"""

class Config(object):
  pass

def get_config(is_train):
  config = Config()
  if is_train:
    #data preparation
    config.gpu_id = '0'
    config.device = 'cuda'
    config.batch_size_data = 64
    config.iters = 2000
    config.im_size = [28, 28]
    #loop in epoch
    config.batch_size_perEpoch = 128
    config.lr = 1e-4
    config.number_workers=16
    config.epochs = 1000
    config.channel = 1
    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
  else:
    config.batch_size = 10
    config.im_size = [28, 28]
    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
  return config
