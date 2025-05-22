#!/bin/bash

export DATASET_DIR="/joint_training/cosmos-predict1"
# export GT_VIDEO_DIR="tokenbench_v1_fps_30_240p_direct_square"
# export GT_VIDEO_CLIPS_DIR="tokenbench_v1_fps_30_240p_direct_square"
export CHECKPOINT_DIR="/joint_training/runimage/imaginaire4/ckpts"


export model_name=$1
export pt_name=$2
export strategy=$3
export avg_rate=$4
export tokenizer_type=$5
export GT_VIDEO_CLIPS_DIR=$6
export OUTPUT_VIDEO_CLIPS_DIR="reconstructions_${GT_VIDEO_CLIPS_DIR}"
export overlap_window=0 # Assign 0 to disable overlapping

cd /joint_training/cosmos-predict1

export strategy="elbo"
# export GT_VIDEO_CLIPS_DIR="davis_square_clip"
# export OUTPUT_VIDEO_CLIPS_DIR="reconstructions_${GT_VIDEO_CLIPS_DIR}"
echo "Start evaluating strategy: $strategy, avg_rate: $avg_rate, tokenizer_type: $tokenizer_type, overlap_window: $overlap_window"
bash exp_scripts/full_eval_called.sh

export strategy="psnr_elbo"
# export GT_VIDEO_CLIPS_DIR="tokenbench_v1_fps_30_240p_direct_square_clip"
# export OUTPUT_VIDEO_CLIPS_DIR="reconstructions_${GT_VIDEO_CLIPS_DIR}"
echo "Start evaluating strategy: $strategy, avg_rate: $avg_rate, tokenizer_type: $tokenizer_type, overlap_window: $overlap_window"
bash exp_scripts/full_eval_called.sh

