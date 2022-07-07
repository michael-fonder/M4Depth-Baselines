#!/bin/bash
# eval DeepV2D on the selected scene on the Tartanair dataset. Choice: neighborhood or oldtown

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/TartanAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/deepv2d-kitti"

cd "$SCRIPT_DIR/../../DeepV2D"

rm -r "data/tartanair"

case "$1" in

    "neighborhood")
        cp -r "data/tartanair-full/urban/test_data/neighborhood" "data/tartanair/"
        ;;

    "oldtown")
        cp -r "data/tartanair-full/urban/test_data/oldtown" "data/tartanair/"
        ;;

    *)
        echo "ERROR: Wrong or no scene argument supplied. Choice: neighborhood or oldtown"
        exit 1
        ;;
esac


python evaluation/eval_tartanair_causal.py --model="$LOG_DIR/kitti.ckpt" --dataset_dir="$DATASET_DIR" --env="urban"