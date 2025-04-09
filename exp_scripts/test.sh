

#PBS -N adaptive_cosmos
#PBS -S /bin/bash
#PBS -l select=1:ncpus=12:mem=90gb:ngpus=4:host=cvml01

nvidia-smi
cd ~/cosmos-predict1
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cosmos

timestamp=$(date +%Y%m%d%H%M%S)
use_latent_tokens=False
head_count=8
batch_size=32
grad_accum_iter=4
num_latent_tokens=64
if [ "$use_latent_tokens" = "True" ]; then
    host_id=29503
else
    host_id=29501
fi

export PYTHONPATH=$(pwd)
export OUTPUT_ROOT=checkpoints
export CUDA_HOME=$CONDA_PREFIX
export TORCH_HOME=/home/qiyuan/.cache/torch/hub

torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:${host_id} \
    -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=ADV8x16x16_256p_ImageNet_Posttrain \
    job.name=QY_ADV8x16x16_256p_ImageNet_${timestamp}_latent${use_latent_tokens}_numLatent${num_latent_tokens} \
    model.config.network.use_latent_tokens=${use_latent_tokens} \
    model.config.network.vit_config.num_attention_heads=${head_count} \
    dataloader_train.batch_size=${batch_size} \
    dataloader_val.batch_size=${batch_size} \
    trainer.grad_accum_iter=${grad_accum_iter} \
    model.config.network.num_latent_tokens=${num_latent_tokens} \
