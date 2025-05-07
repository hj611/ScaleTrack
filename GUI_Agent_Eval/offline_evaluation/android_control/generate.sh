#!/bin/bash

eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate qwen2_val
echo "env activated"
echo $CONDA_DEFAULT_ENV

export http_proxy=http://10.229.18.23:3128
export https_proxy=http://10.229.18.23:3128

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/ScaleTrack/GUI_Agent_Eval/offline_evaluation/android_control

LLM_PATH='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/cache/output/agent/qwen2vl/qwen2vl-fix/stage2_add_all_prompt_fix_callback_l1/checkpoint-16000'
output_file='./logs_0430/16000_low.json'
model='QWEN2VL_Llama_prompt_l1'

CUDA_VISIBLE_DEVICES=2 python ./generate.py --model_path $LLM_PATH \
    --model_type $model \
    --input_file ./data/test.json \
    --output_file $output_file \
    --level low