#!/bin/bash
# Train RNN depth pose on the Mid-Air dataset.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/MidAir")"
LOG_DIR="$(realpath "$SCRIPT_DIR/../../trained-networks/rnn_depth_pose-midair")"

cd "$SCRIPT_DIR/../../RNN_depth_pose"

records="data/data/midair/train_data"

python3 main.py --dataset="midair" --dataset_dir="$DATASET_DIR" --records_dir="$records" --checkpoint_dir="$LOG_DIR" --num_epochs=50