#!/bin/bash
# eval PWCDC-Net on the selected scene on the Tartanair dataset. Choice: gascola or winter

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/TartanAir")"
savepath="$(realpath "$SCRIPT_DIR/../../trained-networks/pwcdc-net-midair")"

cd "$SCRIPT_DIR/../../PWCDCNet"

dataset=$1;

case "$dataset" in

    "gascola")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair"
        fi
        db_seq_len=""
        data="data/tartanair/unstructured/test_data/gascola"
        dataset="tartanair"
        ;;

    "winter")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair"
        fi
        db_seq_len=""
        data="data/tartanair/unstructured/test_data/seasonsforest_winter"
        dataset="tartanair"
        ;;

    *)
        echo "ERROR: Wrong or no scene argument supplied. Choice: gascola or winter"
        exit 1
        ;;
esac

python main.py --mode=eval --dataset="$dataset" $db_seq_len --arch_depth=6 --ckpt_dir="$savepath" --records="$data"