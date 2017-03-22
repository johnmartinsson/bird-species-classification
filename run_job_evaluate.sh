#!/bin/bash

source ./venv/bin/activate
python3 -u evaluate.py --root_path=/disk/martinsson-spring17/birdClef2016Whole/ \
                       --model_name=resnet_18 \
                       --weight_path=weights/2017_01_18_19:27:53_resnet_18.h5
