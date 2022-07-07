#!/bin/bash
# Test ST-CLSTM on the Mid-Air dataset.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/MidAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/st-clstm-midair"

cd "$SCRIPT_DIR/../../ST-CLSTM"

records="splits/midair/test_data/"

python3 evaluate.py --dataset="midair" --test_files="$records" --root_path "$DATASET_DIR" --batch_size=4 --seq_len=8 --loadckpt="$LOG_DIR/ResNet18_checkpoints_small_50" --w=384 --h=384