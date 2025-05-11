#!/usr/bin/env bash
set -e

source TokenBench/env.sh

# Use eval to properly activate conda in the script
eval "$(conda shell.bash hook)"
conda activate cosmos

# 1. Detect GPUs
NGPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NGPUS GPUs"

# 2. Gather files and compute splits
FILES=( "${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR}"/*.mp4 )
TOTAL=${#FILES[@]}
PER_GPU=$(( (TOTAL + NGPUS - 1) / NGPUS ))
echo "Total clips: $TOTAL; ~ ${PER_GPU} per GPU"

# 3. Parallel reconstruction
for (( i=0; i<NGPUS; i++ )); do
  SUBDIR="${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR}/subset_${i}"
  mkdir -p "$SUBDIR"

  START=$(( i * PER_GPU ))
  END=$(( START + PER_GPU ))
  (( END > TOTAL )) && END=$TOTAL

  for (( j=START; j<END; j++ )); do
    ln -s "${FILES[j]}" "$SUBDIR/"
  done

  CUDA_VISIBLE_DEVICES="$i" python3 -m cosmos_predict1.tokenizer.inference.video_cli \
      --video_pattern "${SUBDIR}/*.mp4" \
      --checkpoint "${CHECKPOINT_DIR}/${model_name}/${pt_name}.pt" \
      --output_dir "${DATASET_DIR}/${OUTPUT_VIDEO_CLIPS_DIR}/${model_name}_${pt_name}_${strategy}${avg_rate}" \
      --temporal_window 33 \
      --mode torch \
      --strategy "${strategy}" \
      --avg_rate "${avg_rate}" \
      --tokenizer_type "${tokenizer_type}" &
done

wait

# Change directory and activate conda environment properly
cd TokenBench
# Use eval to properly activate conda in the script
eval "$(conda shell.bash hook)"
conda activate tokenbench

# Run the metrics evaluation
CUDA_VISIBLE_DEVICES=0 python3 -m token_bench.metrics_cli \
    --gtpath ${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR} \
    --targetpath ${DATASET_DIR}/${OUTPUT_VIDEO_CLIPS_DIR}/${model_name}_${pt_name}_${strategy}${avg_rate} \
    --mode all

# Return to original directory
cd ..

# Clean up the intermediate subset directories
for (( i=0; i<NGPUS; i++ )); do
  SUBDIR="${DATASET_DIR}/${GT_VIDEO_CLIPS_DIR}/subset_${i}"
  rm -rf "$SUBDIR"
  echo "Removed temporary subset directory: $SUBDIR"
done
