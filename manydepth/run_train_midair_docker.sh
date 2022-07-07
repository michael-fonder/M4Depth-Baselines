#!/bin/bash

cd MidAir-baseline-methods/manydepth/
python -m manydepth.train --data_path=/workspace/datasets/MidAir --log_dir="logs/run-$1/" --model_name="manydepth" --split="m4depth-MidAir" --dataset="midair" --height=384 --width=384 --min_depth=0.1 --max_depth=100.0 --batch_size=4 --num_epochs=25 --num_layers=50
