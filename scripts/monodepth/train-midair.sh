#!/bin/bash
# Train monodepth on the Mid-Air dataset

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/MidAir")"
LOG_DIR="$(realpath "$SCRIPT_DIR/../../trained-networks/monodepth-midair")"

cd "$SCRIPT_DIR/../../monodepth"

python3 monodepth_main.py --mode=train --dataset=midair --data_path="$DATASET_DIR/" \
                          --filenames_file="splits/midair/train_stereo_files.txt" \
                          --input_height=384 --input_width=384 --batch_size=18 --encoder=resnet50 \
                          --log_directory="$LOG_DIR" \
                          --model_name=""