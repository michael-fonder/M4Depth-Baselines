#!/bin/bash
# eval monodepth on the selected scene on the Tartanair dataset. Choice: gascola or winter

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/TartanAir")"
savepath="$(realpath "$SCRIPT_DIR/../../trained-networks/pwcdc-net-kitti")"

cd "$SCRIPT_DIR/../../PWCDCNet"

dataset=$1;

case "$dataset" in

    "neighborhood")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/kitti"
        fi
        db_seq_len=""
        data="data/tartanair/urban/test_data/neighborhood"
        dataset="tartanair"
        ;;

    "oldtown")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/kitti"
        fi
        db_seq_len=""
        data="data/tartanair/urban/test_data/oldtown"
        dataset="tartanair"
        ;;

    *)
        echo "ERROR: Wrong or no scene argument supplied. Choice: neighborhood or oldtown"
        exit 1
        ;;
esac

python main.py --mode=eval --dataset="$dataset" $db_seq_len --arch_depth=6 --ckpt_dir="$savepath" --records="$data"