#!/bin/bash

rate=${1:-100}
device=${2:-0}

post="DBMLCL-WebCOCO-${rate}"
printFreq=100

mode="DBMLCL"
dataset="DualWebCOCO"
ann_file="annotations/web_coco/train_${rate}_subset0.json,annotations/web_coco/train_${rate}_subset1.json"

pretrainedModel="None"
resumeModel="None"
evaluate="False"

epochs=20
startEpoch=0
stepEpoch=15

batchSize=8
lr=1e-5
momentum=0.9
weightDecay=0
warmupEpoch=1

cropSize=448
scaleSize=512 
workers=2

cleanEpoch=6
theta1=0.8
theta2=0.01
alpha=0.5
prototypeNum=10
instLossWeight=0.01
protoLossWeight=0.05


OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$device python DBMLCL.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \
    --dataset ${dataset} \
    --ann_file ${ann_file} \
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --warmupEpoch ${warmupEpoch} \
    --cropSize ${cropSize} \
    --scaleSize ${scaleSize} \
    --workers ${workers} \
    --cleanEpoch ${cleanEpoch} \
    --theta1 ${theta1} \
    --theta2 ${theta2} \
    --alpha ${alpha} \
    --prototypeNum ${prototypeNum} \
    --instLossWeight ${instLossWeight} \
    --protoLossWeight ${protoLossWeight}
