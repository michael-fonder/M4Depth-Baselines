#!/bin/bash
# Test monodepth2 on the Mid-Air dataset

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/MidAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/monodepth2-midair"

cd "$SCRIPT_DIR/../../monodepth2"

python evaluate_depth.py --data_path="$DATASET_DIR" --model_name="monodepth2" \
                                   --eval_split="m4depth_midair" \
                                   --dataset="midair" \
                                   --height=384 --width=384 \
                                   --min_depth=0.1 --max_depth=80.0 \
                                   --batch_size=4 --num_epochs=25 \
                                   --load_weights_folder="$LOG_DIR/monodepth2/models/weights_24" \
                                   --eval_mono