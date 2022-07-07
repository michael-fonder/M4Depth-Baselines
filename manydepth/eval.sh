#!/bin/bash

cd MidAir-baseline-methods/manydepth/
python -m manydepth.evaluate_depth --data_path=/workspace/datasets/MidAir --log_dir="logs/run-$1/" --model_name="manydepth" --eval_split="m4depth-MidAir" --dataset="midair" --height=384 --width=384 --min_depth=0.1 --max_depth=100.0 --batch_size=4 --num_epochs=25 --num_layers=50 --load_weights_folder=logs/run-$1/manydepth/models/weights_24 --eval_mono --export_pics
