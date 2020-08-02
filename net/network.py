# -*- coding: utf-8 -*-

from net.bcnn import BCNN
from net.vgg import VGG16
from net.resnet import ResNet18, ResNet50


def make_network(net):
    if net == 'vgg16':
        Net = VGG16
        feature_dim = 512
    elif net == 'bcnn':
        Net = BCNN
        feature_dim = 512
    elif net == 'resnet18':
        Net = ResNet18
        feature_dim = 512
    elif net == 'resnet50':
        Net = ResNet50
        feature_dim = 2048
    else:
        raise AssertionError('please specify the correct net in \{bcnn, vgg16, resnet18, resnet50 \}')
    return Net, feature_dim
