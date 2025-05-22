#!/usr/bin/env bash
set -e

source TokenBench/env.sh

# Use eval to properly activate conda in the script
# eval "$(conda shell.bash hook)"
# conda activate cosmos

# 1. Detect GPUs
# NGPUS=$(nvidia-smi --list-gpus | wc -l)
# echo "Detected $NGPUS GPUs"
NGPUS=8



# Change directory and activate conda environment properly
cd TokenBench
# Use eval to properly activate conda in the script
# eval "$(conda shell.bash hook)"
# conda activate tokenbench

modes=(psnr lpips fvd)

# Run the metrics evaluation
# loop gpu 0 to 2 and set mode separately to psnr, lpips, and fvd
for avg_rate in 0.25 0.5 0.75; do
  OUTPUT_PATH="${DATASET_DIR}/${OUTPUT_VIDEO_CLIPS_DIR}/optimization_results_${avg_rate}"
  for idx in "${!modes[@]}"; do
    # gpu should be $idx + 3
    gpuidx=$((idx + 3))
    echo "Start evaluating ${modes[$idx]} using gpu $idx"
    CUDA_VISIBLE_DEVICES=$gpuidx python3 -m token_bench.metrics_cli \
        --gtpath ${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR} \
        --targetpath $OUTPUT_PATH \
        --mode        "${modes[$idx]}" &
        # --mode all
  done
done

wait

# Return to original directory
# cd ..

# Clean up the intermediate subset directories
# for (( i=0; i<NGPUS; i++ )); do
#   SUBDIR="${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR}/subset_${i}"
#   rm -rf "$SUBDIR"
#   echo "Removed temporary subset directory: $SUBDIR"
# done
