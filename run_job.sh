#!/bin/bash

source ./venv/bin/activate
python3 -u ./train_model.py --weight_path=$1 --history_path=$2\
				--train_path=$3 \
				--valid_path=$4 \
				--lock_file=$5\
				--first_epoch=$6
