# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from utils.utils import print_to_logfile


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


def std_loss(logits, labels, indices, T_k, epoch, memory_pool, eps=0.1):
    ce_losses = label_smoothing_cross_entropy(logits, labels, epsilon=eps, reduction='none')

    # in the first T_k epochs, train with the entire training set
    if epoch < T_k:
        # print('using naive CE', end=' <--- ')
        return ce_losses, ce_losses

    # after T_k epochs, start dividing training set into clean / uncertain / irrelevant
    ind_loss_sorted = torch.argsort(ce_losses.data)
    num_remember = torch.nonzero(ce_losses < ce_losses.mean()).shape[0]

    # print(' ---> {:2d}'.format(num_remember), end=', ')
    stats_log_path = 'stats/drop_n_reuse_stats_epoch{:03d}.csv'.format(epoch+1)
    print_to_logfile(stats_log_path, '{:03d}'.format(num_remember), init=False, end=',')

    ind_clean = ind_loss_sorted[:num_remember]
    ind_forget = ind_loss_sorted[num_remember:]
    logits_clean = logits[ind_clean]
    labels_clean = labels[ind_clean]

    if ind_forget.shape[0] > 1:
        # for samples with high loss
        #   high loss, high std --> mislabeling 
        #   high loss, low std  --> irrelevant category
        indices_forget = indices[ind_forget]
        logits_forget = logits[ind_forget]
        pred_distribution = F.softmax(logits_forget, dim=1)
        batch_std = pred_distribution.std(dim=1)

        flag = F.softmax(logits_clean, dim=1).std(dim=1).mean().item()
        # print('{:.5f}'.format(flag), end='*****')
        
        batch_std_sorted, ind_std_sorted = torch.sort(batch_std.data, descending=True)
        ind_split = split_set(batch_std_sorted, flag)
        if ind_split is None:
            ind_split = -1 
        # print('{} == {}'.format(batch_std_sorted, ind_split), end=' ---> ')

        # uncertain could be either mislabeled or hard example
        ind_uncertain = ind_std_sorted[:(ind_split+1)]

        # print('{:2d}/{:2d}'.format(ind_split+1, logits.shape[0] - num_remember), end=' <--- ')
        print_to_logfile(stats_log_path,
                         '{:03d},{:03d}'.format(ind_split+1, logits.shape[0] - num_remember - ind_split - 1),
                         init=False, end='\n')

        logits_mislabeled = logits_forget[ind_uncertain]
        indices_mislabeled = indices_forget[ind_uncertain]
        labels_mislabeled = memory_pool.most_prob_labels[indices_mislabeled].to(logits_mislabeled.device)

        logits_final = torch.cat((logits_clean, logits_mislabeled), dim=0)
        labels_final = torch.cat((labels_clean, labels_mislabeled), dim=0)
    else:
        # print('{:2d}/{:2d}'.format(0, logits.shape[0] - num_remember), end=' <--- ')
        print_to_logfile(stats_log_path, '{:03d},{:03d}'.format(0, logits.shape[0] - num_remember), init=False, end='\n')
        logits_final = logits_clean
        labels_final = labels_clean
    std_losses = label_smoothing_cross_entropy(logits_final, labels_final, epsilon=eps, reduction='none')
    return std_losses, ce_losses


def split_set(x, flag):
    # split set based in interval
    # x shape is (N), x is sorted in descending
    assert (x > 0).all()
    if x.shape[0] == 1:
        return None
    tmp = (x < flag).nonzero()
    if tmp.shape[0] == 0:
        return None
    else:
        return tmp[0, 0] - 1


def label_smoothing_cross_entropy(logit, label, epsilon=0.1, reduction='none'):
    N = label.size(0)
    C = logit.size(1)
    smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(label, dim=1).cpu(), value=1-epsilon)
    if logit.is_cuda:
        smoothed_label = smoothed_label.cuda()

    log_logit = F.log_softmax(logit, dim=1)
    losses = -torch.sum(log_logit * smoothed_label, dim=1)  # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / N
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def cot_std_loss(logits1, logits2, labels, indices, T_k, epoch, memory_pool1, memory_pool2, eps=0.1):
    ce_losses1 = label_smoothing_cross_entropy(logits1, labels, epsilon=eps, reduction='none')  # (N,)
    ce_losses2 = label_smoothing_cross_entropy(logits2, labels, epsilon=eps, reduction='none')  # (N,)

    # in the first T_k epochs, train with the entire training set
    if epoch < T_k:
        # print('using naive CE', end=' <--- ')
        return ce_losses1, ce_losses1, ce_losses2, ce_losses2

    # after T_k epochs, start dividing training set into clean / uncertain / irrelevant
    ind_loss_sorted1 = torch.argsort(ce_losses1.data)
    ind_loss_sorted2 = torch.argsort(ce_losses2.data)
    num_remember1 = torch.nonzero(ce_losses1 < ce_losses1.mean()).shape[0]
    num_remember2 = torch.nonzero(ce_losses2 < ce_losses2.mean()).shape[0]

    # print(' ---> {:2d}, {:2d}'.format(num_remember1, num_remember1), end=', ')
    stats_log_path1 = 'stats/net1_drop_n_reuse_stats_epoch{:03d}.csv'.format(epoch + 1)
    stats_log_path2 = 'stats/net2_drop_n_reuse_stats_epoch{:03d}.csv'.format(epoch + 1)
    print_to_logfile(stats_log_path1, '{:03d}'.format(num_remember1), init=False, end=',')
    print_to_logfile(stats_log_path2, '{:03d}'.format(num_remember2), init=False, end=',')

    ind_clean1 = ind_loss_sorted1[:num_remember1]
    ind_clean2 = ind_loss_sorted2[:num_remember2]
    ind_forget1 = ind_loss_sorted1[num_remember1:]
    ind_forget2 = ind_loss_sorted2[num_remember2:]
    logits_clean1 = logits1[ind_clean2]
    logits_clean2 = logits2[ind_clean1]
    labels_clean1 = labels[ind_clean2]
    labels_clean2 = labels[ind_clean1]

    logits_final1 = logits_clean1
    logits_final2 = logits_clean2
    labels_final1 = labels_clean1
    labels_final2 = labels_clean2

    if ind_forget1.shape[0] > 1:
        # for samples with high loss
        #   high loss, high std --> mislabeling
        #   high loss, low std  --> irrelevant category
        # indices_forget1 = indices[ind_forget1]
        logits_forget1 = logits1[ind_forget1]
        pred_distribution1 = F.softmax(logits_forget1, dim=1)
        batch_std1 = pred_distribution1.std(dim=1)

        flag1 = F.softmax(logits_clean1, dim=1).std(dim=1).mean().item()
        # print('{:.5f}'.format(flag), end='*****')

        batch_std_sorted1, ind_std_sorted1 = torch.sort(batch_std1.data, descending=True)
        ind_split1 = split_set(batch_std_sorted1, flag1)
        if ind_split1 is None:
            ind_split1 = -1
            # print('{} == {}'.format(batch_std_sorted, ind_split), end=' ---> ')

        # uncertain could be either mislabeled or hard example
        ind_uncertain1 = ind_std_sorted1[:(ind_split1 + 1)]

        # print('{:2d}/{:2d}'.format(ind_split1 + 1, logits1.shape[0] - num_remember1), end=' <--- ')
        print_to_logfile(stats_log_path1,
                         '{:03d},{:03d}'.format(ind_split1+1, logits1.shape[0] - num_remember1 - ind_split1 - 1))

        ind_mislabeled1 = ind_forget1[ind_uncertain1]
        logits_mislabeled2 = logits2[ind_mislabeled1]
        indices_mislabeled2 = indices[ind_mislabeled1]
        labels_mislabeled2 = memory_pool2.most_prob_labels[indices_mislabeled2].to(logits_mislabeled2.device)

        logits_final2 = torch.cat((logits_final2, logits_mislabeled2), dim=0)
        labels_final2 = torch.cat((labels_final2, labels_mislabeled2), dim=0)
    if ind_forget2.shape[0] > 1:
        # for samples with high loss
        #   high loss, high std --> mislabeling
        #   high loss, low std  --> irrelevant category
        # indices_forget2 = indices[ind_forget2]
        logits_forget2 = logits2[ind_forget2]
        pred_distribution2 = F.softmax(logits_forget2, dim=1)
        batch_std2 = pred_distribution2.std(dim=1)

        flag2 = F.softmax(logits_clean2, dim=1).std(dim=1).mean().item()
        # print('{:.5f}'.format(flag), end='*****')

        batch_std_sorted2, ind_std_sorted2 = torch.sort(batch_std2.data, descending=True)
        ind_split2 = split_set(batch_std_sorted2, flag2)
        if ind_split2 is None:
            ind_split2 = -1
            # print('{} == {}'.format(batch_std_sorted, ind_split), end=' ---> ')

        # uncertain could be either mislabeled or hard example
        ind_uncertain2 = ind_std_sorted2[:(ind_split2 + 1)]

        # print('{:2d}/{:2d}'.format(ind_split2 + 1, logits2.shape[0] - num_remember2), end=' <--- ')
        print_to_logfile(stats_log_path2,
                         '{:03d},{:03d}'.format(ind_split2+1, logits2.shape[0] - num_remember2 - ind_split2 - 1))

        ind_mislabeled2 = ind_forget2[ind_uncertain2]
        logits_mislabeled1 = logits1[ind_mislabeled2]
        indices_mislabeled1 = indices[ind_mislabeled2]
        labels_mislabeled1 = memory_pool1.most_prob_labels[indices_mislabeled1].to(logits_mislabeled1.device)

        logits_final1 = torch.cat((logits_final1, logits_mislabeled1), dim=0)
        labels_final1 = torch.cat((labels_final1, labels_mislabeled1), dim=0)
    else:
        # print('{:2d}/{:2d}, {:2d}/{:2d}'.format(0, logits1.shape[0] - num_remember1,
        #                                         0, logits2.shape[0] - num_remember2), end=' <--- ')
        print_to_logfile(stats_log_path1, '{:03d},{:03d}'.format(0, logits1.shape[0] - num_remember1))
        print_to_logfile(stats_log_path2, '{:03d},{:03d}'.format(0, logits2.shape[0] - num_remember2))

    losses1 = label_smoothing_cross_entropy(logits_final1, labels_final1, epsilon=eps, reduction='none')
    losses2 = label_smoothing_cross_entropy(logits_final2, labels_final2, epsilon=eps, reduction='none')
    return losses1, ce_losses1, losses2, ce_losses2
