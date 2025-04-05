#!/bin/bash

debug=$1 # choose between true or false

num_encoder_layers=10
num_decoder_layers=10
use_latent=False

if [ "$debug" = "true" ]; then
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m  cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=Adaptive_Tokenize1_ADV8x16x16_256p_ImageNet \
    model.config.network.vit_config.num_encoder_layers=${num_encoder_layers} \
    model.config.network.vit_config.num_decoder_layers=${num_decoder_layers} \
    model.config.network.vit_config.use_latent=${use_latent} \
    job.name=imagenet_enc${num_encoder_layers}_dec${num_decoder_layers}_latent${use_latent} \
    job.project=posttraining
else
    torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=Adaptive_Tokenize1_ADV8x16x16_256p_ImageNet \
    model.config.network.vit_config.num_encoder_layers=${num_encoder_layers} \
    model.config.network.vit_config.num_decoder_layers=${num_decoder_layers} \
    model.config.network.vit_config.use_latent=${use_latent} \
    job.name=imagenet_enc${num_encoder_layers}_dec${num_decoder_layers}_latent${use_latent} \
    job.project=posttraining
fi






