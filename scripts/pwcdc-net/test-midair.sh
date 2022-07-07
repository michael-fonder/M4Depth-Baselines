#!/bin/bash
# eval monodepth on the selected scene on the Tartanair dataset. Choice: gascola or winter

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/TartanAir")"
savepath="$(realpath "$SCRIPT_DIR/../../trained-networks/pwcdc-net-midair")"

cd "$SCRIPT_DIR/../../PWCDCNet"

dataset="midair"

db_seq_len=""
data="data/midair/test_data"

python main.py --mode=eval --dataset="$dataset" $db_seq_len --arch_depth=6 --ckpt_dir="$savepath" --records="$data"