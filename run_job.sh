#!/bin/bash

source ./venv/bin/activate
python3 -u ./train_model.py --weight_path=$1 \
        --history_path=$2 \
				--lock_file=$3 \
        --config_file=$4 \
				--first_epoch=$5
