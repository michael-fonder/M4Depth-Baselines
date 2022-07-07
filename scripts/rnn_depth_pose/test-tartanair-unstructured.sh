#!/bin/bash
# Test RNN depth pose on the selected scene on the Tartanair dataset. Choice: gascola or winter

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/TartanAir")"
LOG_DIR="$(realpath "$SCRIPT_DIR/../../trained-networks/rnn_depth_pose-midair")"

cd "$SCRIPT_DIR/../../RNN_depth_pose"

case "$1" in

    "gascola")
        records="data/data/tartanair/unstructured/test_data/gascola"
        ;;

    "winter")
        records="data/data/tartanair/unstructured/test_data/seasonsforest_winter"
        ;;

    *)
        echo "ERROR: Wrong or no scene argument supplied. Choice: gascola or winter"
        exit 1
        ;;
esac

python3 eval.py --dataset="tartanair" --eval_set_dir="$DATASET_DIR" --records_dir="$records" --restore_path="$LOG_DIR" --checkpoint_dir="$LOG_DIR"