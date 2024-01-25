#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 ${SCRIPT_DIR}/../src/main.py --batch-size 64 --lr-patience 13 --lr 0.01 --network TestNet --data-config base --nepochs 300 --validation 0 --target-fps 15 --window-size 16 --exp-name muellnet --labeled-start  --lr-min 1e-5 --flip-prob 0.5 