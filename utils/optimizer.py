# -*- coding: utf-8 -*-

import math
import numpy as np
import torch.optim as optim


def make_optimizer(params, lr, weight_decay, opt='Adam'):
    if opt == 'Adam':
        print('| Adam Optimizer is used ... ')
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif opt == 'SGD':
        print('| SGD Optimizer is used ... ')
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)


def get_lr_from_optimizer(optimizer):
    lr = -1
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr
