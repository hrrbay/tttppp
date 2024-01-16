for seed in 1 2 3; do
    python3 src/main.py --batch-size 10 --lr-patience 6 --lr 0.1 --network r2plus1d_18 --data-config r3d_18 --nepochs 50 --labeled-start --validation-vid 4 --target-fps 15 --window-size 16 --seed $seed --exp-name r2plus1d_sgd_01_val4_scratch --flip-prob 0.5
done