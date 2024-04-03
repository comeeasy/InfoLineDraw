#! /bin/bash

######## 240403 실험 ########
#
# 앞서 수행한 실험의 결과가 좋지 않음. 여전히 음영이 남음.
# 이를 해결하기 위해 학습적으로는 geom loss를 안사용할 것임.
# 
# train 에서 --use_geom 0 으로 셋팅
# 
############################

export CUDA_VISIBLE_DEVICES=0

dataset_root="/home/joono/media2/workspace/InfoLineDraw/train_dset"

for arg in "$@"
do
case $arg in
    train)
    ### Train ###
    python train_pl.py \
        --name 240403 \
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
        --use_geom 0 \
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