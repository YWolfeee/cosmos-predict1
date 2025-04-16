#PBS -N adaptive_cosmos
#PBS -S /bin/bash
#PBS -l select=1:ncpus=12:mem=90gb:ngpus=4:host=cvml01

nvidia-smi
cd ~/cosmos-predict1
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cosmos

timestamp=$(date +%Y%m%d%H%M%S)
head_count=8
rate_startegy="static"
batch_size=2
grad_accum_iter=1
host_id=29501

export PYTHONPATH=$(pwd)
export OUTPUT_ROOT=checkpoints
export CUDA_HOME=$CONDA_PREFIX
export TORCH_HOME=/home/qiyuan/.cache/torch/hub

WANDB_MODE=offline torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${host_id} \
    -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=ADV8x16x16_256p_ImageNet_Posttrain \
    job.name=QY_ADV8x16x16_256p_ImageNet_${timestamp}_numLatent${num_latent_tokens} \
    model.config.network.vit_config.num_attention_heads=${head_count} \
    dataloader_train.batch_size=${batch_size} \
    dataloader_val.batch_size=${batch_size} \
    trainer.grad_accum_iter=${grad_accum_iter} \
    model.config.network.rate_strategy=${rate_startegy}