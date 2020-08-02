# -*- coding: utf-8 -*-

import time
import argparse


def show_params(params):
    data_path = params['data_base']
    step = params['step']
    print('|-----------------------------------------------------')
    if step == 1:
        print('| Step 1: Fine-tune FC layer only ... ')
    elif step == 2:
        print('| Step 2: Train the entire network... ')
    elif step == 0:
        print('| Step 0: One step training ...')
    else:
        raise AssertionError()
    print('|-----------------------------------------------------')
    print('| Training Config : ')
    print('| net             : {}'.format(params['net']))
    print('| learning_rate   : {}'.format(params['lr']))
    print('| weight_decay    : {}'.format(params['weight_decay']))
    print('| batch_size      : {}'.format(params['batch_size']))
    print('| epochs          : {}'.format(params['epochs']))
    print('| num of classes  : {}'.format(params['n_classes']))
    print('|-----------------------------------------------------')
    print('| warm-up epochs  : {}'.format(params['warmup_epochs']))
    print('| memory length   : {}'.format(params['memory_length']))
    print('| epsilon         : {}'.format(params['eps']))
    print('|-----------------------------------------------------')
    print('| data_path  : {}'.format(data_path))
    print('| log_file   : {}'.format(params['log']))
    print('|-----------------------------------------------------')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--coteach', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--n_classes', type=int, default=200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--log', type=str, default='logfile.txt')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--mem_length', type=int, default=5)
    parser.add_argument('--eps', type=float, default=0.0)
    
    args = parser.parse_args()

    assert args.step == 0 or args.step == 1 or args.step == 2, 'wrong step argument!'

    if args.step == 0:
        logfile = 'log/{}'.format(args.log)
    else:
        logfile = 'log/step_{}-{}'.format(args.step, args.log)

    config = {
        'data_base': args.dataset,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'net': args.net,
        'step': args.step,
        'n_classes': args.n_classes,
        'resume': args.resume,
        'log': logfile,
        'warmup_epochs': args.warmup_epochs,
        'memory_length': args.mem_length,
        'eps': args.eps,
    }

    show_params(config)

    if args.coteach:
        from cot_train import CoteachingTrainer as Trainer
    else:
        from train import Trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end-start))
