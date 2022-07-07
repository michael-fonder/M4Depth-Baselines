#!/bin/bash
# eval monodepth on the selected scene on the Tartanair dataset. Choice: neighborhood or oldtown

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/TartanAir")"
LOG_DIR="$(realpath "$SCRIPT_DIR/../../trained-networks/monodepth-kitti")"

cd "$SCRIPT_DIR/../../monodepth"

case "$1" in

    "neighborhood")
        DATA="splits/tartanair/unstructured/neighborhood"
        ;;

    "oldtown")
        DATA="splits/tartanair/unstructured/oldtown"
        ;;

    *)
        echo "ERROR: Wrong or no scene argument supplied. Choice: neighborhood or oldtown"
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR/outputs-$1"

python3 monodepth_main.py --mode=test --dataset=midair --data_path="$DATASET_DIR/" \
                          --filenames_file="$DATA/stereo_samples.txt" \
                          --input_height=384 --input_width=512 --batch_size=18 \
                          --log_directory="$LOG_DIR" \
                          --encoder=resnet50 --model_name "" \
                          --output_directory="$LOG_DIR/outputs-$1"

python3 evaluate_kitti.py --split=tartanair --predicted_disp_path="$LOG_DIR/outputs-$1/disparities.npy" --gt_path="$DATA" --db_path="$DATASET_DIR"