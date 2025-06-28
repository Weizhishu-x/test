#!/usr/bin/env bash

set -x

EXP_DIR=./exp/eval
BATCH_SIZE=8
RESUME=./exps/r50_deformable_detr_bs16_source/best_checkpoint.pth
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size ${BATCH_SIZE} \
    --resume ${RESUME} \
    --eval
    ${PY_ARGS}
