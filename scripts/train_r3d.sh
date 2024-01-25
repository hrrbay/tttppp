#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 ${SCRIPT_DIR}/../src/main.py --batch-size 32 --lr-patience 6 --lr 0.01 --network r3d_18 --data-config r3d_18 --nepochs 200 --fixed-seq-len 200 --validation 0.1 --target-fps 15 --window-size 16 --flip-prob 0.5 --lr-patience 6 --exp-name r3d_flip_val01_sgd --gpu 0 --lr-min 1e-5
