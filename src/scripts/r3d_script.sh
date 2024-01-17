for seed in 1 2 3 5 6 7 8; do
    python3 src/main.py --batch-size 32 --lr-patience 6 --lr 0.01 --network r3d_18 --data-config r3d_18 --nepochs 50 --fixed-seq-len 200 --validation 0.1 --target-fps 15 --window-size 16 --seed $seed --flip-prob 0.5 --lr-patience 6 --exp-name r3d_flip_val01_sgd --gpu -1
done
