#!/bin/bash
# eval manydepth on the selected scene on the Tartanair dataset. Choice: neighborhood or oldtown

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/TartanAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/manydepth-kitti"

cd "$SCRIPT_DIR/../../manydepth"

case "$1" in

    "neighborhood")
        cp "splits/m4depth-tartanair-urban/test_files-neighborhood.txt" "splits/m4depth-tartanair-urban/test_files.txt"
        ;;

    "oldtown")
        cp "splits/m4depth-tartanair-urban/test_files-oldtown.txt" "splits/m4depth-tartanair-urban/test_files.txt"
        ;;

    *)
        echo "ERROR: Wrong or no scene argument supplied. Choice: neighborhood or oldtown"
        exit 1
        ;;
esac

python -m manydepth.evaluate_depth --data_path="$DATASET_DIR" --model_name="manydepth" \
                                   --eval_split="m4depth-tartanair-urban" \
                                   --dataset="tartanair" \
                                   --min_depth=0.001 --max_depth=80.0 \
                                   --batch_size=4 --num_epochs=25 \
                                   --load_weights_folder="$LOG_DIR/KITTI_HR" --eval_mono