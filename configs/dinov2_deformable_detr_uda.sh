#!/usr/bin/env bash

set -x
BACKBONE='facebook/dinov2-base'
EXP_DIR=./exps/xView2DOTA_uda/dinov2_projector_4_2_1_0.5_lora_b2_base
# PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --backbone ${BACKBONE} \
    --batch_size 2 \
    --projector_scale 4.0 2.0 1.0 0.5 \
    # --resume ./exps/xView2DOTA/best_set/best_checkpoint.pth \
    # --eval \
    # --visualize \
    # ${PY_ARGS}