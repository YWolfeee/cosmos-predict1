#!/usr/bin/env bash
set -e

source env.sh

# Determine the target path based on overlap_window value
if [ "$overlap_window" -eq 0 ]; then
    target_path="${DATASET_DIR}/${OUTPUT_VIDEO_CLIPS_DIR}/${model_name}_${pt_name}_${strategy}${avg_rate}"
else
    target_path="${DATASET_DIR}/${OUTPUT_VIDEO_CLIPS_DIR}/${model_name}_${pt_name}_${strategy}${avg_rate}_overlap${overlap_window}"
fi

python3 -m token_bench.metrics_cli \
    --gtpath ${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR} \
    --targetpath ${target_path} \
    --mode all