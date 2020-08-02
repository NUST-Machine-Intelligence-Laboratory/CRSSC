# -*- coding: utf-8 -*-

import torch
from utils.metrics import AverageMeter, accuracy


def evaluate(dataloader, model, topk=(1,)):
    """

    :param dataloader:
    :param model:
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_accuracy = AverageMeter()
    test_accuracy.reset()

    with torch.no_grad():
        for sample in dataloader:
            if len(sample) == 2:
                x, y = sample
            else:
                x, y, _ = sample
            x = x.cuda()
            y = y.cuda()
            logits = model(x)

            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))

    return test_accuracy.avg
