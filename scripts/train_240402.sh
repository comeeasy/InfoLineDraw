#! /bin/bash


export CUDA_VISIBLE_DEVICES=0

dataset_root="/home/joono/media2/workspace/InfoLineDraw/train_dset"

for arg in "$@"
do
case $arg in
    train)
    ### Train ###
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
    ;;
    test)
    ### Test ###
    python test.py \
        --name lightning_logs/version_0/checkpoints \
        --dataroot examples/test \
        --n_blocks 5 \
        --size 512 \
        --cuda \
        --n_cpu 16 \
        --num_classes 107 \
    ;;
    *)
    echo "Invalid argument: $arg"
    echo "Usage: $0 [train] [test]"
    ;;
esac
done