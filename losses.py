#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:40:54 2019

@author: shariba
"""
import torch
from config import get_config
config = get_config(is_train=True)


def ncc(x1, x2):
    epsilon = 0.000001
    x1 = (x1 - torch.mean(x1))/ (torch.std(x1) + epsilon)
    x2 = (x2 - torch.mean(x2))/ (torch.std(x2)+ epsilon)
    xncc =  torch.sum(x1 * x2)/len(x1)
    return -torch.mean(xncc)


def ncc2(x1, x2):
    prod = torch.mean((x1 - torch.mean(x1) ) *  (x2 - torch.mean(x2) ))
    stds = torch.std(x1) * torch.std(x2)
    if torch.equal(stds, torch.zeros(1)[0]):
        return torch.zeros(1)
    else:
        prod /= stds
#        torch.div(prod,stds)
        # minimize negative of ncc (since we use adam minimizer) that is maximize ncc 
        return -prod

def nccLoss(x1, x2):
    device = config.device
    """ batch image should be divided by size, ZNCC = 1-0.5 (x_cap - y_cap)^2
        xncc ==> 1 - Zncc = 1, lowest corr, 1-zncc = 0, highest correlation
        value (0, 1===> should go below with minimization techniques like Adam)
    """
    epsilon = 0.000001 # avoid division by 0
    x1 = (x1 - torch.mean(x1))/ (torch.std(x1) + epsilon)
    x2 = (x2 - torch.mean(x2))/ (torch.std(x2) + epsilon)
    xncc = (0.5*torch.mean((x1-x2)**2)).to(device)
    return xncc

def nccLoss_RegL2(x1, x2, u):
    """ batch image should be divided by size, ZNCC = 1-0.5 (x_cap - y_cap)^2
        xncc ==> 1 - Zncc = 1, lowest corr, 1-zncc = 0, highest correlation
        value (0, 1===> should go below with minimization techniques like Adam)
    """
    epsilon = 0.000001 # avoid division by 0
    lambda_1 = 0.01 # TODO: put this as a learnable parameter
    x1 = (x1 - torch.mean(x1))/ (torch.std(x1) + epsilon)
    x2 = (x2 - torch.mean(x2))/ (torch.std(x2) + epsilon)
    # added an l2-norm to regularize for smooth deformatin field
    xncc = 0.5*torch.mean((x1-x2)**2) + lambda_1 * torch.norm(u)
    return xncc


def nccLoss_clipVersion(x1, x2):
    """ 
    Description: same as SSD version of ZNCC but with slight modification
    this loss can be interesting for multi-modal cases
    """
    device = config.device
    std_thr = torch.tensor(0.3).to(device)
    x1 = torch.abs(x1 - torch.mean(x1))/ torch.clamp(torch.std(x1), std_thr)
    x2 = torch.abs(x2 - torch.mean(x2))/ torch.clamp(torch.std(x2), std_thr)
    xncc = 0.5*torch.mean((x1-x2)**2)
    return xncc
