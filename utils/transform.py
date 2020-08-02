# -*- coding: utf-8 -*-

import torchvision


def make_transform(phase='train', output_size=448):
    if phase == 'train':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=output_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=output_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    elif phase == 'test':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=output_size),
            torchvision.transforms.CenterCrop(size=output_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        raise AssertionError('Not implemented yet')
    return transform
