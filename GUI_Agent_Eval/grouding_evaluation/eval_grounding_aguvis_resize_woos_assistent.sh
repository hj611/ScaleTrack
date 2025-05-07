#!/bin/bash

eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/Anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# conda activate aguvis
conda activate qwen2vl

echo "env activated"
echo $CONDA_DEFAULT_ENV

export http_proxy=http://10.229.18.23:3128
export https_proxy=http://10.229.18.23:3128
export WANDB_API_KEY=82007ed41298807b1d9a801e540996878db64f87

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/ScaleTrack/GUI_Agent_Eval/grouding_evaluation

# 定义不同的checkpoint和对应的GPU设置
declare -A checkpoints=(
    ["checkpoint-20000"]="0,1"
    ["checkpoint-28000"]="2,3"
    ["checkpoint-32983"]="4,5"
    ["checkpoint-24000"]="6,7"
    # ["checkpoint-20000"]="0"
    # ["checkpoint-28000"]="1"
    # ["checkpoint-32983"]="2"
    # ["checkpoint-43977"]="3"
)

screenspot_imgs="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/data/screen/ScreenSpot/eval/images"
screenspot_test="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/data/screen/ScreenSpot/eval"

# 循环遍历每个checkpoint
for checkpoint in "${!checkpoints[@]}"; do
    # LLM_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/cache/output/agent/qwen2vl/qwen2vl-fix/stage1_add/$checkpoint"
    LLM_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/cache/output/agent/qwen2vl/qwen2vl-fix/stage1_add_72b/$checkpoint"
    GPU_DEVICES="${checkpoints[$checkpoint]}"
    
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python eval/screenspot_test_aguivs_resize_woos_assistent.py --model_path $LLM_PATH \
        --screenspot_imgs $screenspot_imgs \
        --screenspot_test $screenspot_test \
        --task all \
        --mode ScaleTrackG \
        > "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/ScaleTrack/GUI_Agent_Eval/grouding_evaluation/logs/stage1_add_${checkpoint}_micro.log" 2>&1 &
    
    echo "Finished evaluation for $checkpoint with GPUs $GPU_DEVICES"
done

# 等待所有后台进程完成
wait

python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/code/tools/tools/gpu_util.py
