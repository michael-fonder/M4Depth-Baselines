#!/bin/bash
# Test RNN depth pose on the Mid-Air dataset.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/MidAir")"
LOG_DIR="$(realpath "$SCRIPT_DIR/../../trained-networks/rnn_depth_pose-midair")"

cd "$SCRIPT_DIR/../../RNN_depth_pose"

records="data/data/midair/test_data"

python3 eval.py --dataset="midair" --eval_set_dir="$DATASET_DIR" --records_dir="$records" --restore_path="$LOG_DIR" --checkpoint_dir="$LOG_DIR"