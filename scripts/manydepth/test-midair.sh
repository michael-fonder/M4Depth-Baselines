#!/bin/bash
# Test manydepth on the Mid-Air dataset

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/MidAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/manydepth-midair"

cd "$SCRIPT_DIR/../../manydepth"

python -m manydepth.evaluate_depth --data_path="$DATASET_DIR" --model_name="manydepth" \
                                   --eval_split="m4depth-MidAir" \
                                   --dataset="midair" \
                                   --min_depth=0.1 --max_depth=80.0 \
                                   --batch_size=4 --num_epochs=25 --num_layers=50 \
                                   --load_weights_folder="$LOG_DIR/manydepth/models/weights_24" \
                                   --eval_mono