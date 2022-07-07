#!/bin/bash

cd MidAir-baseline-methods/manydepth/
python -m manydepth.train --data_path=/workspace/datasets/TartanAir --log_dir="logs/run-$1/" --model_name="manydepth" --split="m4depth-tartanair" --dataset="tartanair" --height=384 --width=512 --min_depth=0.1 --max_depth=100.0 --batch_size=4 --num_epochs=25 --num_layers=50 --load_weights_folder="logs/run-$1/manydepth/models/weights_9/"
