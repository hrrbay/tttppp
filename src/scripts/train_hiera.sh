#!/bin/bash

python main.py \
--network HieraB \
--data-config hiera \
--window-size 16\
 --model-config ../config/hiera.yaml \
--train-config ../config/train.yaml \
--labeled-start \
--target-fps 15 \
--validation 0.2

