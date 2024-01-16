for seed in 1 2 3; do
    python3 src/main.py --batch-size 32 --lr-patience 6 --lr 0.01 --network r3d_18 --data-config r3d_18 --nepochs 10 --pretrained --labeled-start --validation 0.1 --target-fps 15 --window-size 16 --seed $seed
done