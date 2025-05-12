#!/usr/bin/env bash
set -e

source env.sh

CUDA_VISIBLE_DEVICES=2 python3 -m token_bench.metrics_cli \
    --gtpath /mnt/rdata8/qy_dataset/TokenBench/tokenbench_v1_fps_30_240p_direct_square \
    --targetpath /mnt/rdata8/qy_dataset/TokenBench/reconstruction_tokenbench_v1_fps_30_240p_direct_square/elastic_0.020 \
    --mode all