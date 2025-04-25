#!/bin/bash

debug=$1 # choose between true or false

extra_kwargs="
model.config.network.vit_config.switch_rotary_to_1d=0.5
model.config.network.rate_strategy='unibin'
"


if [ "$debug" = "true" ]; then
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=ADV4x8x8_256p_HDVILA_Posttrain $extra_kwargs
else
    torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=ADV4x8x8_256p_HDVILA_Posttrain $extra_kwargs
fi






