# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from utils.indexedImageFolder import IndexedImageFolder
import numpy as np
import random
from PIL import ImageFile

from net.network import make_network

from utils.utils import print_to_console, print_to_logfile
from utils.transform import make_transform
from utils.metrics import AverageMeter, accuracy
from utils.optimizer import make_optimizer, get_lr_from_optimizer
from utils.eval import evaluate
from utils.container import Queue
from loss import cot_std_loss

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


class CoteachingTrainer(object):
    def __init__(self, config):
        # Config
        self._config = config
        self._epochs = config['epochs']
        self._step = config['step']
        self._logfile = config['log']
        self._n_classes = config['n_classes']

        # Network
        if ',' in config['net']:
            net_name_1, net_name_2 = config['net'].split(',')
        else:
            net_name_1, net_name_2 = config['net'], config['net']
        Net1, _ = make_network(net_name_1)
        Net2, _ = make_network(net_name_2)
        if self._step == 0:
            net1 = Net1(n_classes=self._n_classes, pretrained=True, use_two_step=False, fc_init='He')
            net2 = Net2(n_classes=self._n_classes, pretrained=True, use_two_step=False, fc_init='Xavier')
        elif self._step == 1:
            net1 = Net1(n_classes=self._n_classes, pretrained=True, use_two_step=True)
            net2 = Net2(n_classes=self._n_classes, pretrained=True, use_two_step=True)
        elif self._step == 2:
            net1 = Net1(n_classes=self._n_classes, pretrained=False, use_two_step=True)
            net2 = Net2(n_classes=self._n_classes, pretrained=False, use_two_step=True)
        else:
            raise AssertionError('step can only be 0, 1, 2')
        # Move network to cuda
        print('| Number of available GPUs : {} ({})'.format(torch.cuda.device_count(),
                                                            os.environ["CUDA_VISIBLE_DEVICES"]))
        if torch.cuda.device_count() >= 1:
            self._net1 = nn.DataParallel(net1).cuda()
            self._net2 = nn.DataParallel(net2).cuda()
        else:
            raise AssertionError('CPU version is not implemented yet!')

        # Loss Criterion
        self.T_k = config['warmup_epochs']
        if self._step == 1:
            self.T_k = self._epochs

        # Optimizer
        if self._step == 1:
            params_to_optimize1 = self._net1.module.fc.parameters()
            params_to_optimize2 = self._net2.module.fc.parameters()
        else:
            params_to_optimize1 = self._net1.parameters()
            params_to_optimize2 = self._net2.parameters()
        self._optimizer1 = make_optimizer(params_to_optimize1, lr=config['lr'] / 2, weight_decay=config['weight_decay'],
                                          opt='SGD')
        self._optimizer2 = make_optimizer(params_to_optimize2, lr=config['lr'], weight_decay=config['weight_decay'],
                                          opt='SGD')

        self._scheduler1 = optim.lr_scheduler.CosineAnnealingLR(self._optimizer1, T_max=self._epochs, eta_min=0)
        self._scheduler2 = optim.lr_scheduler.CosineAnnealingLR(self._optimizer2, T_max=self._epochs, eta_min=0)

        # metrics
        self._train_loss1 = AverageMeter()
        self._train_loss2 = AverageMeter()
        self._train_accuracy1 = AverageMeter()
        self._train_accuracy2 = AverageMeter()
        self._epoch_train_time = AverageMeter()

        # Dataloader
        train_transform = make_transform(phase='train', output_size=448)
        test_transform = make_transform(phase='test', output_size=448)
        train_data = IndexedImageFolder(os.path.join(config['data_base'], 'train'), transform=train_transform)
        test_data = IndexedImageFolder(os.path.join(config['data_base'], 'val'), transform=test_transform)
        self._train_loader = data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                                             pin_memory=True)
        self._test_loader = data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        print('|-----------------------------------------------------')
        print('| Number of samples in train set : {}'.format(len(train_data)))
        print('| Number of samples in test  set : {}'.format(len(test_data)))
        print('| Number of classes in train set : {}'.format(len(train_data.classes)))
        print('| Number of classes in test  set : {}'.format(len(test_data.classes)))
        print('|-----------------------------------------------------')
        assert len(train_data.classes) == self._n_classes and \
            len(test_data.classes) == self._n_classes, 'number of classes is wrong'

        # Resume or not
        if config['resume']:
            assert os.path.isfile('checkpoint.pth'), 'no checkpoint.pth exists!'
            print('---> loading checkpoint.pth <---')
            checkpoint = torch.load('checkpoint.pth')
            assert self._step == checkpoint['step'], 'step in checkpoint does not match step in argument'
            self._start_epoch = checkpoint['epoch']
            self._best_accuracy1 = checkpoint['best_accuracy1']
            self._best_accuracy2 = checkpoint['best_accuracy2']
            self._best_epoch1 = checkpoint['best_epoch1']
            self._best_epoch2 = checkpoint['best_epoch2']
            self._net1.load_state_dict(checkpoint['state_dict1'])
            self._net2.load_state_dict(checkpoint['state_dict2'])
            self._optimizer1.load_state_dict(checkpoint['optimizer1'])
            self._optimizer2.load_state_dict(checkpoint['optimizer2'])
            self._scheduler1.load_state_dict(checkpoint['scheduler1'])
            self._scheduler2.load_state_dict(checkpoint['scheduler2'])
            self.memory_pool1 = checkpoint['memory_pool1']
            self.memory_pool2 = checkpoint['memory_pool2']
        else:
            print('---> no checkpoint loaded <---')
            if self._step == 2:
                print('---> loading step1_best_epoch.pth <---')
                assert os.path.isfile('model/step1_best_epoch.pth')
                self._net1.load_state_dict(torch.load('model/net1_step1_best_epoch.pth'))
                self._net2.load_state_dict(torch.load('model/net2_step1_best_epoch.pth'))
            self._start_epoch = 0
            self._best_accuracy1 = 0.0
            self._best_accuracy2 = 0.0
            self._best_epoch1 = None
            self._best_epoch2 = None
            self.memory_pool1 = Queue(n_samples=len(train_data), memory_length=config['memory_length'])
            self.memory_pool2 = Queue(n_samples=len(train_data), memory_length=config['memory_length'])
        self._scheduler1.last_epoch = self._start_epoch
        self._scheduler2.last_epoch = self._start_epoch

    def train(self):
        console_header = 'Epoch\tTrain_Loss1\tTrain_Loss2\tTrain_Accuracy1\tTrain_Accuracy2\t' \
                         'Test_Accuracy1\tTest_Accuracy2\tEpoch_Runtime\tLearning_Rate1\tLearning_Rate2'
        print_to_console(console_header)
        print_to_logfile(self._logfile, console_header, init=True)

        for t in range(self._start_epoch, self._epochs):
            epoch_start = time.time()
            self._scheduler1.step(epoch=t)
            self._scheduler2.step(epoch=t)
            # reset average meters
            self._train_loss1.reset()
            self._train_loss2.reset()
            self._train_accuracy1.reset()
            self._train_accuracy2.reset()

            self._net1.train(True)
            self._net2.train(True)
            self.single_epoch_training(t)
            test_accuracy1 = evaluate(self._test_loader, self._net1)
            test_accuracy2 = evaluate(self._test_loader, self._net2)

            lr1 = get_lr_from_optimizer(self._optimizer1)
            lr2 = get_lr_from_optimizer(self._optimizer2)

            if test_accuracy1 > self._best_accuracy1:
                self._best_accuracy1 = test_accuracy1
                self._best_epoch1 = t + 1
                torch.save(self._net1.state_dict(), 'model/net1_step{}_best_epoch.pth'.format(self._step))
            if test_accuracy2 > self._best_accuracy2:
                self._best_accuracy2 = test_accuracy2
                self._best_epoch2 = t + 1
                torch.save(self._net2.state_dict(), 'model/net2_step{}_best_epoch.pth'.format(self._step))

            epoch_end = time.time()
            single_epoch_runtime = epoch_end - epoch_start
            # Logging
            console_content = '{:05d}\t{:10.4f}\t{:10.4f}\t{:14.4f}\t{:14.4f}\t' \
                              '{:13.4f}\t{:13.4f}\t{:13.2f}\t' \
                              '{:13.1e}\t{:13.1e}'.format(t + 1, self._train_loss1.avg, self._train_loss2.avg,
                                                          self._train_accuracy1.avg, self._train_accuracy2.avg,
                                                          test_accuracy1, test_accuracy2,
                                                          single_epoch_runtime, lr1, lr2)
            print_to_console(console_content)
            print_to_logfile(self._logfile, console_content, init=False)

            # save checkpoint
            save_checkpoint({
                'epoch': t + 1,
                'state_dict1': self._net1.state_dict(),
                'state_dict2': self._net2.state_dict(),
                'best_epoch1': self._best_epoch1,
                'best_epoch2': self._best_epoch2,
                'best_accuracy1': self._best_accuracy1,
                'best_accuracy2': self._best_accuracy2,
                'optimizer1': self._optimizer1.state_dict(),
                'optimizer2': self._optimizer2.state_dict(),
                'step': self._step,
                'scheduler1': self._scheduler1.state_dict(),
                'scheduler2': self._scheduler2.state_dict(),
                'memory_pool1': self.memory_pool1,
                'memory_pool2': self.memory_pool2,
            })

        console_content = 'Net1: Best at epoch {}, test accuracy is {}'.format(self._best_epoch1, self._best_accuracy1)
        print_to_console(console_content)
        console_content = 'Net2: Best at epoch {}, test accuracy is {}'.format(self._best_epoch2, self._best_accuracy2)
        print_to_console(console_content)

        # rename log file
        os.rename(self._logfile, self._logfile.replace('.txt', '-{}_{}_{}_{:.4f}_{:.4f}.txt'.format(
            self._config['net'], self._config['batch_size'], self._config['lr'],
            self._best_accuracy1, self._best_accuracy2)))

    def single_epoch_training(self, epoch, log_iter=True, log_freq=200):
        if epoch >= self.T_k:
            stats_log_path1 = 'stats/net1_drop_n_reuse_stats_epoch{:03d}.csv'.format(epoch+1)
            stats_log_path2 = 'stats/net2_drop_n_reuse_stats_epoch{:03d}.csv'.format(epoch+1)
            stats_log_header = 'clean_sample_num,reusable_sample_num,irrelevant_sample_num'
            print_to_logfile(stats_log_path1, stats_log_header, init=True, end='\n')
            print_to_logfile(stats_log_path2, stats_log_header, init=True, end='\n')

        for it, (x, y, indices) in enumerate(self._train_loader):
            s = time.time()

            x = x.cuda()
            y = y.cuda()
            self._optimizer1.zero_grad()
            self._optimizer2.zero_grad()
            logits1 = self._net1(x)
            logits2 = self._net2(x)
            losses1, ce_loss1, losses2, ce_loss2 = \
                cot_std_loss(logits1, logits2, y, indices, self.T_k, epoch,
                             self.memory_pool1, self.memory_pool1, eps=self._config['eps'])
            loss1 = losses1.mean()
            loss2 = losses2.mean()

            self.memory_pool1.update(indices=indices, losses=ce_loss1.detach().data.cpu(),
                                     scores=F.softmax(logits1, dim=1).detach().data.cpu(),
                                     labels=y.detach().data.cpu())
            self.memory_pool1.update(indices=indices, losses=ce_loss2.detach().data.cpu(),
                                     scores=F.softmax(logits2, dim=1).detach().data.cpu(),
                                     labels=y.detach().data.cpu())

            train_accuracy1 = accuracy(logits1, y, topk=(1,))
            train_accuracy2 = accuracy(logits2, y, topk=(1,))

            self._train_loss1.update(loss1.item(), losses1.size(0))
            self._train_loss2.update(loss2.item(), losses1.size(0))
            self._train_accuracy1.update(train_accuracy1[0], x.size(0))
            self._train_accuracy2.update(train_accuracy2[0], x.size(0))

            loss1.backward()
            loss2.backward()
            self._optimizer1.step()
            self._optimizer2.step()

            e = time.time()
            self._epoch_train_time.update(e - s, 1)
            if (log_iter and (it + 1) % log_freq == 0) or (it + 1 == len(self._train_loader)):
                console_content = 'Epoch:[{:03d}/{:03d}]  Iter:[{:04d}/{:04d}]  ' \
                                  'Train Accuracy1 :[{:6.2f}]  Train Accuracy2 :[{:6.2f}]  ' \
                                  'Loss1:[{:4.4f}]  Loss2:[{:4.4f}]  ' \
                                  'Iter Runtime:[{:6.2f}]'.format(epoch + 1, self._epochs, it + 1,
                                                                  len(self._train_loader),
                                                                  self._train_accuracy1.avg, self._train_accuracy2.avg,
                                                                  self._train_loss1.avg, self._train_loss2.avg,
                                                                  self._epoch_train_time.avg)
                print_to_console(console_content)
