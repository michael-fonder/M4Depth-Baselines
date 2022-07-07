#!/bin/bash
# eval manydepth on the selected scene on the Tartanair dataset. Choice: gascola or winter

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET_DIR="$SCRIPT_DIR/../../datasets/TartanAir"
LOG_DIR="$SCRIPT_DIR/../../trained-networks/manydepth-midair"

cd "$SCRIPT_DIR/../../manydepth"

case "$1" in

    "gascola")
        cp "splits/m4depth-tartanair-unstructured/test_files-gascola.txt" "splits/m4depth-tartanair-unstructured/test_files.txt"
        ;;

    "winter")
        cp "splits/m4depth-tartanair-unstructured/test_files-winter.txt" "splits/m4depth-tartanair-unstructured/test_files.txt"
        ;;

    *)
        echo "ERROR: Wrong or no scene argument supplied. Choice: gascola or winter"
        exit 1
        ;;
esac

python -m manydepth.evaluate_depth --data_path="$DATASET_DIR" --model_name="manydepth" \
                                   --eval_split="m4depth-tartanair-unstructured" \
                                   --dataset="tartanair" \
                                   --min_depth=0.1 --max_depth=80.0 \
                                   --batch_size=4 --num_epochs=25 --num_layers=50 \
                                   --load_weights_folder="$LOG_DIR/manydepth/models/weights_24" \
                                   --eval_mono