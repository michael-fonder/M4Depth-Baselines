#!/bin/bash
# Train ST-CLSTM on the Mid-Air dataset.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/MidAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/st-clstm-midair"

cd "$SCRIPT_DIR/../../ST-CLSTM"

records="data/data/midair/train_data"

python3 train.py --dataset="midair" --trainlist_path="$records" --root_path="$DATASET_DIR" --batch_size=3 --epochs 50 --logdir="$LOG_DIR/" --checkpoint_dir="$LOG_DIR/"