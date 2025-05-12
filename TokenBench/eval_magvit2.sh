#!/usr/bin/env bash
set -e

source env.sh

python3 -m token_bench.metrics_cli \
    --gtpath /mnt/rdata8/qy_dataset/TokenBench/tokenbench_v1_fps_30_240p_direct_square \
    --targetpath /mnt/rdata8/qy_dataset/TokenBench/reconstruction_tokenbench_v1_fps_30_240p_direct_square/Open-MAGVIT2_256p \
    --mode all