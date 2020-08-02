# -*- coding: utf-8 -*-

import torch
import numpy as np


class Queue(object):
    def __init__(self, n_samples, memory_length=5):
        super().__init__()
        self.n_samples = n_samples
        self.memory_length = memory_length
        # the item in content_dict is as follows:
        #   dict {
        #         key: 'pred', value: [(pred class index, pred probability), ...];
        #         key: 'loss', value: [ loss, ... ];
        #         key: 'most_prob_label', value: predicted label with highest accumulated probability
        #        }
        self.content = np.array([
            {'pred': [], 'loss': [], 'label': -1} for i in range(n_samples)
        ])
        self.most_prob_labels = torch.Tensor([-1 for i in range(n_samples)]).long()

    def update(self, indices, losses, scores, labels):
        probs, preds = scores.max(dim=1)
        for i in range(indices.shape[0]):
            if len(self.content[indices[i].item()]['pred']) >= self.memory_length:
                self.content[indices[i].item()]['pred'].pop(0)
                self.content[indices[i].item()]['loss'].pop(0)
            self.content[indices[i].item()]['pred'].append((preds[i].item(), probs[i].item()))

            try:
                self.content[indices[i].item()]['loss'].append(losses[i].item())
            except:
                print(indices.shape, losses.shape)
                raise AssertionError()

            self.content[indices[i].item()]['label'] = labels[i].item()

        for i in range(indices.shape[0]):
            tmp = {}
            most_prob_label = -1
            highest_prob = 0
            for pred_idx, pred_prob in self.content[indices[i].item()]['pred']:
                if pred_idx not in tmp:
                    tmp[pred_idx] = pred_prob
                else:
                    tmp[pred_idx] += pred_prob
                if highest_prob < tmp[pred_idx]:
                    highest_prob = tmp[pred_idx]
                    most_prob_label = pred_idx
            self.most_prob_labels[indices[i].item()] = most_prob_label
