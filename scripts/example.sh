#! /bin/bash

export CUDA_VISIBLE_DEVICES=1

img_dir="/home/joono/InfoLineDraw/pseudo_img_sketch_dataset/imgs"
sketch_dir="/home/joono/InfoLineDraw/pseudo_img_sketch_dataset/sketches"
depth_dir="/home/joono/InfoLineDraw/pseudo_img_sketch_dataset/depths"

for arg in "$@"
do
case $arg in
    depth)
    ### Generate depth maps ###
    cd ./BoostingMonocularDepth
    python run.py --Final --max_res 3000 --data_dir $img_dir --output_dir $depth_dir  --depthNet 0
    ;;
    train)
    ### Train ###
    cd .
    python train.py \
        --name 240322_2309_test \
        --dataroot $img_dir \
        --depthroot $depth_dir \
        --root2 $sketch_dir \
        --cuda \
        --batchSize 3 \
        --save_epoch_freq 20 \
        --n_epochs 100 \
        --log_int 10 \
        --size 512 \
        --n_cpu 8 \
        --n_blocks 6 \
        --load_size 574 \
        --crop_size 512 \
        --num_classes 107
    ;;
    test)
    ### Test ###
    python test.py \
        --name 240322_2309_test \
        --dataroot examples/test \
        --n_blocks 6 \
        --size 512 \
        --cuda \
        --n_cpu 16 \
    ;;
    *)
    echo "Invalid argument: $arg"
    echo "Usage: $0 [depth] [train] [test]"
    ;;
esac
done