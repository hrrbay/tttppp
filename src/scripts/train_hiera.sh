#!/bin/bash
for seed in 1 2 3 4 5 6 7 8 9 10; do
  python ../main.py \
  --network HieraB \
  --data-config hiera \
  --window-size 16\
  --model-config ../config/hiera.yaml \
  --batch-size 32 \
  --checkpoint-freq 5 \
  --exp-name hiera_last \
  --fixed-seq-len 200 \
  --flip-prob 0.5 \
  --gpu 1 \
  --labeled-start \
  --lr 0.01 \
  --lr-factor 3 \
  --lr-min 0.00001 \
  --lr-patience 6 \
  --momentum 0.9 \
  --nepochs 50 \
  --seed 1 \
  --src-fps 120 \
  --target-fps 15 \
  --use-poses false \
  --validation 0.1 \
  --weight-decay 0.0002 \
  --window-size 16
done
