#!/usr/bin/env bash
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PYTHONPATH}:$PWD"

export CUDA_VISIBLE_DEVICES=0

export DATA_BASE='web-bird'
export N_CLASSES=200
export NET='bcnn'

# --- Config ---
export WARMUP_EPOCHS=5
export MEM_LENGTH=5
export EPS=0.2
# --- Config ---

export LOGFILE="${DATA_BASE}-warmup_${WARMUP_EPOCHS}-memlen_${MEM_LENGTH}-eps_${EPS}.txt"

mkdir -p model
mkdir -p log
mkdir -p stats

if [[ ${NET} != 'bcnn' ]]; then
    python main.py --dataset ${DATA_BASE} \
                   --lr 1e-2 \
                   --batch_size 32 \
                   --weight_decay 3e-4 \
                   --epochs 100 \
                   --net ${NET} \
                   --n_classes ${N_CLASSES} \
                   --step 0 \
                   --log ${LOGFILE} \
                   --warmup_epochs ${WARMUP_EPOCHS} \
                   --mem_length ${MEM_LENGTH} \
                   --eps ${EPS}
else
    python main.py --dataset ${DATA_BASE} \
                   --lr 1 \
                   --batch_size 64 \
                   --weight_decay 1e-8 \
                   --epochs 80 \
                   --net ${NET} \
                   --n_classes ${N_CLASSES} \
                   --step 1 \
                   --log ${LOGFILE} \
                   --eps ${EPS}
    sleep 120s
    python main.py --dataset ${DATA_BASE} \
                   --lr 1e-2 \
                   --batch_size 64 \
                   --weight_decay 1e-4 \
                   --epochs 80 \
                   --net ${NET} \
                   --n_classes ${N_CLASSES} \
                   --step 2 \
                   --log ${LOGFILE} \
                   --warmup_epochs ${WARMUP_EPOCHS} \
                   --mem_length ${MEM_LENGTH} \
                   --eps ${EPS}
fi
