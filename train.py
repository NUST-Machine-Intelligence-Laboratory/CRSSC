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
from loss import std_loss
from utils.container import Queue

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


class Trainer(object):
    def __init__(self, config):
        # Config
        self._config = config
        self._epochs = config['epochs']
        self._step = config['step']
        self._logfile = config['log']
        self._n_classes = config['n_classes']

        # Network
        Net, feature_dim = make_network(config['net'])

        if self._step == 0:
            net = Net(n_classes=self._n_classes, pretrained=True, use_two_step=False)
        elif self._step == 1:
            net = Net(n_classes=self._n_classes, pretrained=True, use_two_step=True)
        elif self._step == 2:
            net = Net(n_classes=self._n_classes, pretrained=False, use_two_step=True)
        else:
            raise AssertionError('step can only be 0, 1, 2')
        # Move network to cuda
        print('| Number of available GPUs : {} ({})'.format(torch.cuda.device_count(),
                                                            os.environ["CUDA_VISIBLE_DEVICES"]))
        if torch.cuda.device_count() >= 1:
            self._net = nn.DataParallel(net).cuda()
        else:
            raise AssertionError('CPU version is not implemented yet!')

        # Loss Criterion
        self.T_k = config['warmup_epochs']
        if self._step == 1:
            self.T_k = self._epochs

        # Optimizer
        if self._step == 1:
            params_to_optimize = self._net.module.fc.parameters()
        else:
            params_to_optimize = self._net.parameters()
        self._optimizer = make_optimizer(params_to_optimize, lr=config['lr'], weight_decay=config['weight_decay'],
                                         opt='SGD')

        self._scheduler = optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._epochs, eta_min=0)
        # metrics
        self._train_loss = AverageMeter()
        self._train_accuracy = AverageMeter()
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
            self._best_accuracy = checkpoint['best_accuracy']
            self._best_epoch = checkpoint['best_epoch']
            self._net.load_state_dict(checkpoint['state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            self._scheduler.load_state_dict(checkpoint['scheduler'])
            self.memory_pool = checkpoint['memory_pool']
        else:
            print('---> no checkpoint loaded <---')
            if self._step == 2:
                print('---> loading step1_best_epoch.pth <---')
                assert os.path.isfile('model/step1_best_epoch.pth')
                self._net.load_state_dict(torch.load('model/step1_best_epoch.pth'))
            self._start_epoch = 0
            self._best_accuracy = 0.0
            self._best_epoch = None
            self.memory_pool = Queue(n_samples=len(train_data), memory_length=config['memory_length'])
        self._scheduler.last_epoch = self._start_epoch

    def train(self):
        console_header = 'Epoch\tTrain_Loss\tTrain_Accuracy\tTest_Accuracy\tEpoch_Runtime\tLearning_Rate'
        print_to_console(console_header)
        print_to_logfile(self._logfile, console_header, init=True)

        for t in range(self._start_epoch, self._epochs):
            epoch_start = time.time()
            self._scheduler.step(epoch=t)
            # reset average meters
            self._train_loss.reset()
            self._train_accuracy.reset()

            self._net.train(True)
            self.single_epoch_training(t)
            test_accuracy = evaluate(self._test_loader, self._net)

            lr = get_lr_from_optimizer(self._optimizer)

            if test_accuracy > self._best_accuracy:
                self._best_accuracy = test_accuracy
                self._best_epoch = t + 1
                torch.save(self._net.state_dict(), 'model/step{}_best_epoch.pth'.format(self._step))
                # print('*', end='')
            epoch_end = time.time()
            single_epoch_runtime = epoch_end - epoch_start
            # Logging
            console_content = '{:05d}\t{:10.4f}\t{:14.4f}\t{:13.4f}\t{:13.2f}\t{:13.1e}'.format(
                t + 1, self._train_loss.avg, self._train_accuracy.avg, test_accuracy, single_epoch_runtime, lr)
            print_to_console(console_content)
            print_to_logfile(self._logfile, console_content, init=False)

            # save checkpoint
            save_checkpoint({
                'epoch': t + 1,
                'state_dict': self._net.state_dict(),
                'best_epoch': self._best_epoch,
                'best_accuracy': self._best_accuracy,
                'optimizer': self._optimizer.state_dict(),
                'step': self._step,
                'scheduler': self._scheduler.state_dict(),
                'memory_pool': self.memory_pool,
            })

        console_content = 'Best at epoch {}, test accuracy is {}'.format(self._best_epoch, self._best_accuracy)
        print_to_console(console_content)

        # rename log file, stats files and model
        os.rename(self._logfile, self._logfile.replace('.txt', '-{}_{}_{}_{:.4f}.txt'.format(
            self._config['net'], self._config['batch_size'], self._config['lr'], self._best_accuracy)))

    def single_epoch_training(self, epoch, log_iter=True, log_freq=100):
        if epoch >= self.T_k:
            stats_log_path = 'stats/drop_n_reuse_stats_epoch{:03d}.csv'.format(epoch+1)
            stats_log_header = 'clean_sample_num,reusable_sample_num,irrelevant_sample_num'
            print_to_logfile(stats_log_path, stats_log_header, init=True, end='\n')
        for it, (x, y, indices) in enumerate(self._train_loader):
            s = time.time()

            x = x.cuda()
            y = y.cuda()
            self._optimizer.zero_grad()
            logits = self._net(x)
            losses, ce_loss = std_loss(logits, y, indices, self.T_k, epoch, self.memory_pool,
                                       eps=self._config['eps'])
            loss = losses.mean()

            self.memory_pool.update(indices=indices, losses=ce_loss.detach().data.cpu(),
                                    scores=F.softmax(logits, dim=1).detach().data.cpu(),
                                    labels=y.detach().data.cpu())

            train_accuracy = accuracy(logits, y, topk=(1,))

            self._train_loss.update(loss.item(), x.size(0))
            self._train_accuracy.update(train_accuracy[0], x.size(0))

            loss.backward()
            self._optimizer.step()

            e = time.time()
            self._epoch_train_time.update(e-s, 1)
            if (log_iter and (it+1) % log_freq == 0) or (it+1 == len(self._train_loader)):
                console_content = 'Epoch:[{0:03d}/{1:03d}]  Iter:[{2:04d}/{3:04d}]  ' \
                                  'Train Accuracy :[{4:6.2f}]  Loss:[{5:4.4f}]  ' \
                                  'Iter Runtime:[{6:6.2f}]'.format(epoch + 1, self._epochs, it + 1,
                                                                   len(self._train_loader),
                                                                   self._train_accuracy.avg,
                                                                   self._train_loss.avg, self._epoch_train_time.avg)
                print_to_console(console_content)
