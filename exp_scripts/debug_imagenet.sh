#!/bin/bash

debug=$1 # choose between true or false

# add timestampe to the end of job.name
timestamp=$(date +%Y%m%d%H%M%S)
experiment="ADV8x16x16_256p_ImageNet_Posttrain"

extra_kwargs="
model.config.network.vit_config.switch_rotary_to_1d=0.5
model.config.network.rate_strategy='unibin'
"


if [ "$debug" = "true" ]; then
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=$experiment job.name=VIT_for_imagenet_1x8x8_${timestamp} $extra_kwargs
else
    torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=$experiment job.name=VIT_for_imagenet_1x8x8_${timestamp} $extra_kwargs
fi






