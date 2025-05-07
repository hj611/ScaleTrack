#!/bin/bash

eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/Anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate qwen25vl
echo "env activated"
echo $CONDA_DEFAULT_ENV

export http_proxy=http://10.229.18.23:3128
export https_proxy=http://10.229.18.23:3128

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/ScaleTrack/GUI_Agent_Eval/offline_evaluation/android_control

response_path='./logs_0430/16000_low.json'
log_path='./logs_0430/metrics_16000_low.json'
model_type='AGUVIS'

python ./eval.py --response_path $response_path \
    --log_path $log_path \
    --model_type $model_type 