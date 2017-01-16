#!/bin/bash

source ./venv/bin/activate
python3 -u ./train_model.py --weight_path=$1 --history_path=$2\
				--train_path=$3 \
				--valid_path=$4 \
                                --noise_path=$5 \
				--lock_file=$6 \
                                --model_name=$7 \
				--first_epoch=$8
