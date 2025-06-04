#!/bin/bash

eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/Anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# conda activate aguvis
conda activate qwen25vl

echo "env activated"
echo $CONDA_DEFAULT_ENV

export http_proxy=http://10.229.18.23:3128
export https_proxy=http://10.229.18.23:3128
export WANDB_API_KEY=82007ed41298807b1d9a801e540996878db64f87

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/ScaleTrack/GUI_Agent_Eval/grouding_evaluation

# 定义不同的checkpoint和对应的GPU设置
declare -A checkpoints=(
    ["checkpoint-8000"]="0"
    ["checkpoint-16000"]="1"
    ["checkpoint-24000"]="2"
    ["checkpoint-32000"]="3"
    ["checkpoint-40000"]="4"
    ["checkpoint-43977"]="5"
     
)

screenspot_imgs="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/data/screen/ScreenSpot/eval/images"
screenspot_test="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/data/screen/ScreenSpot/eval"

# 循环遍历每个checkpoint
for checkpoint in "${!checkpoints[@]}"; do
    LLM_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/cache/output/agent/qwen25vl-fix/stage1_big/$checkpoint"
    GPU_DEVICES="${checkpoints[$checkpoint]}"
    
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python eval/screenspot_test_qwen25vl.py --model_path $LLM_PATH \
        --screenspot_imgs $screenspot_imgs \
        --screenspot_test $screenspot_test \
        --task all \
        --mode ScaleTrackG \
        > "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/ScaleTrack/GUI_Agent_Eval/grouding_evaluation/logs/qwen25vl/stage1_add_${checkpoint}.log" 2>&1 &
    
    echo "Finished evaluation for $checkpoint with GPUs $GPU_DEVICES"
done
 
