#!/bin/bash
# Test monodepth on the Mid-Air dataset

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/MidAir")"
LOG_DIR="$(realpath "$SCRIPT_DIR/../../trained-networks/monodepth-midair")"

cd "$SCRIPT_DIR/../../monodepth"

mkdir -p "$LOG_DIR/outputs-midair"

python3 monodepth_main.py --mode=test --dataset=midair --data_path="$DATASET_DIR/" \
                          --filenames_file="splits/midair/stereo_samples.txt" \
                          --input_height=384 --input_width=384 --batch_size=18 \
                          --log_directory="$LOG_DIR" \
                          --encoder=resnet50 --model_name "" \
                          --output_directory="$LOG_DIR/outputs-midair"

python3 evaluate_kitti.py --split=midair --predicted_disp_path="$LOG_DIR/outputs-midair/disparities.npy" --gt_path="splits/midair" --db_path="$DATASET_DIR"