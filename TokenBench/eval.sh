#!/usr/bin/env bash
set -e

source env.sh

python3 -m token_bench.metrics_cli \
    --gtpath ${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR} \
    --targetpath ${DATASET_DIR}/${OUTPUT_VIDEO_CLIPS_DIR}/${model_name}_${pt_name}_${strategy}${avg_rate} \
    --mode all