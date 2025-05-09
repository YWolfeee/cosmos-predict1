#!/usr/bin/env bash
set -e

source TokenBench/env.sh

python3 -m token_bench.metrics_cli \
    --gtpath ${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR} \
    --targetpath /mnt/rdata8/qy_dataset/TokenBench/reconstruction_subset_square_v1_fps_30_240p_clip/elastic_0_007 \
    --mode all