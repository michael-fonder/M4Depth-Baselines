#!/bin/bash
# Test ST-CLSTM on the selected scene on the Tartanair dataset. Choice: gascola or winter

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/TartanAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/st-clstm-midair"

cd "$SCRIPT_DIR/../../ST-CLSTM"

case "$1" in

    "gascola")
        records="splits/tartanair/unstructured/test_data/gascola/"
        ;;

    "winter")
        records="splits/tartanair/unstructured/test_data/seasonsforest_winter/"
        ;;

    *)
        echo "ERROR: Wrong or no scene argument supplied. Choice: gascola or winter"
        exit 1
        ;;
esac

python3 evaluate.py --dataset="tartanair" --test_files="$records" --root_path "$DATASET_DIR" --batch_size=4 --seq_len=8 --loadckpt="$LOG_DIR/ResNet18_checkpoints_small_50" --w=512 --h=384