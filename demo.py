# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from net.network import make_network
from utils.eval import evaluate
from utils.transform import make_transform
from utils.indexedImageFolder import IndexedImageFolder


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


if __name__ == '__main__':
    assert torch.cuda.device_count() >= 1, 'This is designed to run on GPU but no GPU is found'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str , required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_classes', type=int, default=200)
    parser.add_argument('--net', type=str, default='bcnn')
    args = parser.parse_args()

    NET, _ = make_network(args.net)
    net = NET(n_classes=args.n_classes, pretrained=False)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(args.model))

    test_transform = make_transform(phase='test', output_size=448)
    test_data = IndexedImageFolder(os.path.join(args.data, 'val'), transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    test_accuracy = evaluate(test_loader, net)
    print('| Test accuracy: {:.4f}'.format(test_accuracy))
