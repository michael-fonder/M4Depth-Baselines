#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/MidAir")"
LOG_DIR="$(realpath "$SCRIPT_DIR/../../trained-networks/pwcdc-net-midair")"

cd "$SCRIPT_DIR/../../PWCDCNet"

records="splits/midair/test_data/"

python main.py --mode=train --dataset="midair" --seq_len=4 --db_seq_len=8 --arch_depth=6 --batch_size=24 --ckpt_dir="$LOG_DIR" --log_dir="$LOG_DIR/summaries" --records="data/midair/train_data" --enable_validation
