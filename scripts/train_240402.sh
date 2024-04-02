#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

img_dir="/home/joono/media2/workspace/InfoLineDraw/dataset/imgs"
sketch_dir="/home/joono/media2/workspace/InfoLineDraw/dataset/vecsktch"
depth_dir="/home/joono/media2/workspace/InfoLineDraw/dataset/depths"

dataset_root="/home/joono/media2/workspace/InfoLineDraw/train_dset"

python train_pl.py \
        --name 240402_train_with_26000 \
        --dataset_root $dataset_root \
        --dataroot "" \
        --depthroot "" \
        --root2 "" \
        --cuda \
        --batchSize 4 \
        --save_epoch_freq 5 \
        --n_epochs 100 \
        --log_int 10 \
        --size 512 \
        --n_cpu 16 \
        --n_blocks 5 \
        --load_size 512 \
        --crop_size 512 \
        --num_classes 107 \
        --lr 0.0001 \