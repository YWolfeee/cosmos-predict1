#!/usr/bin/env bash
set -e

source TokenBench/env.sh

CUDA_VISIBLE_DEVICES=2 python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern "${DATASET_DIR}/${GT_VIDEO_DIR}/*.mp4" \
    --checkpoint "${CHECKPOINT_DIR}/${model_name}/${pt_name}.pt" \
    --output_dir ${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR} \
    --temporal_window 33 \
    --mode torch \
    --strategy ${strategy} \
    --tokenizer_type ${tokenizer_type} \
    --save_clip \
    --output_fps 33 \
    
