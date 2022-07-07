#!/bin/bash
# Train monodepth2 on the Mid-Air dataset

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR
DATASET_DIR="$SCRIPT_DIR/../../datasets/MidAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/monodepth2-midair"

cd "$SCRIPT_DIR/../../monodepth2"

python train.py --data_path="$DATASET_DIR" --log_dir="$LOG_DIR" --model_name="monodepth2" \
                          --split="m4depth-midair" --dataset="midair" \
                          --height=384 --width=384 \
                          --min_depth=0.1 --max_depth=80.0 --batch_size=6 --num_epochs=17