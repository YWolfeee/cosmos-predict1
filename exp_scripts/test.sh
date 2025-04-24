#PBS -N adaptive_cosmos
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml01

nvidia-smi
cd ~/cosmos-predict1
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cosmos

timestamp=$(date +%Y%m%d%H%M%S)
head_count=16
num_decoder_layers=8
num_encoder_layers=8
rate_strategy="unibin"
batch_size=4
grad_accum_iter=4
host_id=29501

export PYTHONPATH=$(pwd)
export OUTPUT_ROOT=checkpoints
export CUDA_HOME=$CONDA_PREFIX
export TORCH_HOME=/home/qiyuan/.cache/torch/hub

torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:${host_id} \
    -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=ADV8x16x16_256p_ImageNet_Posttrain \
    job.name=QY_ADV8x16x16_256p_ImageNet_${timestamp} \
    model.config.network.vit_config.num_attention_heads=${head_count} \
    dataloader_train.batch_size=${batch_size} \
    dataloader_val.batch_size=${batch_size} \
    trainer.grad_accum_iter=${grad_accum_iter} \
    model.config.network.rate_strategy=${rate_strategy} \
    model.config.network.vit_config.num_decoder_layers=${num_decoder_layers} \
    model.config.network.vit_config.num_encoder_layers=${num_encoder_layers} \

# WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${host_id} \
#     -m cosmos_predict1.tokenizer.training.train \
#     --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
#     experiment=ADV8x16x16_256p_ImageNet_Posttrain \
#     job.name=QY_ADV8x16x16_256p_ImageNet_${timestamp} \
#     model.config.network.vit_config.num_attention_heads=${head_count} \
#     dataloader_train.batch_size=${batch_size} \
#     dataloader_val.batch_size=${batch_size} \
#     trainer.grad_accum_iter=${grad_accum_iter} \
#     model.config.network.rate_strategy=${rate_strategy} \
#     model.config.network.vit_config.num_decoder_layers=${num_decoder_layers} \
#     model.config.network.vit_config.num_encoder_layers=${num_encoder_layers} \
#     checkpoint.save_iter=1 \
#     trainer.validation_iter=1 \
