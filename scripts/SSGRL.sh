#!/bin/bash

rate=${1:-100}
deviceIds=${2:-0}
deviceType=${3:-auto}

post="SSGRL-WebCOCO-${rate}"
printFreq=100

mode="Base"
dataset="WebCOCO"
ann_file="annotations/web_coco/train_${rate}.json"

pretrainedModel="None"
resumeModel="None"
evaluate="False"

epochs=20
startEpoch=0
stepEpoch=15

batchSize=8
lr=1e-5
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512 
workers=2


OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${deviceIds} ASCEND_RT_VISIBLE_DEVICES=${deviceIds} python SSGRL.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \
    --dataset ${dataset} \
    --ann_file ${ann_file} \
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --device ${deviceType} \
    --deviceIds ${deviceIds} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --cropSize ${cropSize} \
    --scaleSize ${scaleSize} \
    --workers ${workers}
