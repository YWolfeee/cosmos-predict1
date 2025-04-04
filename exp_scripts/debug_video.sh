#!/bin/bash

debug=$1 # choose between true or false

if [ "$debug" = "true" ]; then
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=Adaptive_Tokenize1_ADV8x16x16_720p_HDVILA
else
    torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=Adaptive_Tokenize1_ADV8x16x16_720p_HDVILA
fi






