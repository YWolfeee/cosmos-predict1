#!/bin/bash

debug=$1 # choose between true or false

experiment="ADV4x8x8_256p_HDVILA_Posttrain"

extra_kwargs="
model.config.network.vit_config.switch_rotary_to_1d=0.25
model.config.network.rate_strategy='unibin'
model.config.network.vit_config.num_encoder_layers=16
model.config.network.vit_config.num_decoder_layers=32
dataloader_train.batch_size=2
dataloader_train.num_workers=6
dataloader_train.dataset.num_video_frames=49
dataloader_val.batch_size=2
dataloader_val.num_workers=6
dataloader_val.dataset.num_video_frames=49
model.config.network.temporal_compression=8
"


if [ "$debug" = "true" ]; then
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=$experiment $extra_kwargs
else
    torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=$experiment $extra_kwargs
fi






