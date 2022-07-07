#!/bin/bash
# Train manydepth on the Mid-Air dataset

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/MidAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/manydepth-midair"

cd "$SCRIPT_DIR/../../manydepth"

python -m manydepth.train --data_path="$DATASET_DIR" --log_dir="$LOG_DIR" --model_name="manydepth" \
                          --split="m4depth-MidAir" --dataset="midair" \
                          --height=384 --width=384 \
                          --min_depth=0.1 --max_depth=80.0 --batch_size=4 --num_epochs=25 --num_layers=50