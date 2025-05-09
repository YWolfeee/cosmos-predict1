#!/usr/bin/env bash
set -e

source TokenBench/env.sh

CUDA_VISIBLE_DEVICES=2 python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern "${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR}/*.mp4" \
    --checkpoint ${CHECKPOINT_DIR}/${model_name}/${pt_name}.pt \
    --output_dir ${DATASET_DIR}/${OUTPUT_VIDEO_CLIPS_DIR}/${model_name}_${pt_name}_${strategy}${avg_rate} \
    --temporal_window 25 \
    --mode torch \
    --strategy ${strategy} \
    --avg_rate ${avg_rate} \
    --tokenizer_type ${tokenizer_type} \
    
