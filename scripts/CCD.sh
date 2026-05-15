#!/bin/bash

rate=${1:-100}
deviceIds=${2:-0}
deviceType=${3:-auto}

post="CCD-WebCOCO-${rate}"
printFreq=50

mode="CCD"
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
weightDecay=1e-4

cropSize=448
scaleSize=512
workers=2

clipModelPath="./pretrained/RN50x64.pt"
lrMult=10
offsetSize=40
coeff=0.8
lossCoeff=0.1
ratio=0.8
updateLabel="True"
infNum=1
globalTemp="True"
localTemp="True"
useConsist=1
bound=4

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${deviceIds} ASCEND_RT_VISIBLE_DEVICES=${deviceIds} python CCD.py \
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
    --workers ${workers} \
    --clipModelPath ${clipModelPath} \
    --lrMult ${lrMult} \
    --offsetSize ${offsetSize} \
    --coeff ${coeff} \
    --lossCoeff ${lossCoeff} \
    --ratio ${ratio} \
    --updateLabel ${updateLabel} \
    --infNum ${infNum} \
    --globalTemp ${globalTemp} \
    --localTemp ${localTemp} \
    --useConsist ${useConsist} \
    --bound ${bound}
