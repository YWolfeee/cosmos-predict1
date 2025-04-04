#PBS -N adaptive_cosmos
#PBS -S /bin/bash
#PBS -l select=1:ncpus=24:mem=180gb:ngpus=4:host=cvml06

nvidia-smi
cd ~/cosmos-predict1
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cosmos

num_encoder_layers = 8
num_decoder_layers = 16

export PYTHONPATH=$(pwd)
export OUTPUT_ROOT=checkpoints
export CUDA_HOME=$CONDA_PREFIX
export TORCH_HOME=/home/qiyuan/.cache/torch/hub

torchrun --nproc_per_node=4 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=mock_Adaptive_Tokenize1_ADV8x16x16_720p_ImageNet \
    model.config.network.config.num_encoder_layers=${num_encoder_layers} \
    model.config.network.config.num_decoder_layers=${num_decoder_layers} \
    job.name=imagenet_enc${num_encoder_layers}_dec${num_decoder_layers}