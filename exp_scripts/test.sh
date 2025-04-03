#PBS -N adaptive_cosmos
#PBS -S /bin/bash
#PBS -l select=1:ncpus=24:mem=180gb:ngpus=4:host=cvml01

nvidia-smi
cd ~/cosmos-predict1
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cosmos

export PYTHONPATH=$(pwd)
export OUTPUT_ROOT=checkpoints
export CUDA_HOME=$CONDA_PREFIX
export TORCH_HOME=/home/qiyuan/.cache/torch/hub

torchrun --nproc_per_node=4 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=Adaptive_Tokenize1_ADV8x16x16_720p_HDVILA