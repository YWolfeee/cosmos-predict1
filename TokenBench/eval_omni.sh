#!/usr/bin/env bash
set -e

source env.sh

python3 -m token_bench.metrics_cli \
    --gtpath /mnt/rdata8/qy_dataset/TokenBench/omni_tokenizer/haotian/gt \
    --targetpath /mnt/rdata8/qy_dataset/TokenBench/omni_tokenizer/haotian/recons \
    --mode all