#!/usr/bin/env bash

set -x
BACKBONE='facebook/dinov2-base'
EXP_DIR=./exps/xView2DOTA_uda/dinov2_projector_1_b4_base
# PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --backbone ${BACKBONE} \
    --batch_size 4 \
    --feature_extraction_layers 2 5 8 11 \
    --projector_scale 1.0 \
