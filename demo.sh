#!/usr/bin/env bash
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PYTHONPATH}:$PWD"

export CUDA_VISIBLE_DEVICES=0

export DATA_BASE='web-bird'
export N_CLASSES=200
export NET='bcnn'

python demo.py --data ${DATA_BASE} --model model/web-bird_bcnn_best_epoch_77.4249.pth --n_classes ${N_CLASSES} --net ${NET}
