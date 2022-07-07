#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$(realpath "$SCRIPT_DIR/../../datasets/MidAir")"
savepath="$(realpath "$SCRIPT_DIR/../../trained-networks/pwcdc-net-kitti")"

rm -r "$savepath"
cp -r "$(realpath "$SCRIPT_DIR/../../trained-networks/pwcdc-net-midair")" "$savepath"

cd "$SCRIPT_DIR/../../PWCDCNet"

if [ ! -d "$savepath/train-midair" ]
then
    mv "$savepath/train" "$savepath/train-midair";
    mv "$savepath/best" "$savepath/train"; 
fi

python finetune-kitti.py --arch_depth=6 --batch_size=16 --ckpt_dir="$savepath" --log_dir="$savepath/summaries" --records=data --enable_validation
