#!/bin/bash
#SBATCH --job-name=1N@quick_tokenizer_eval@P1
#SBATCH --time=02:00:00             # Time limit
#SBATCH --account=dir_cosmos_misc
#SBATCH --partition=batch
#SBATCH --mem-per-gpu=200G 
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8


# 这里根据自己需要决定是不是要指定 output 和 error 文件
#SBATCH --output=logs/debug/%j_%x.out
#SBATCH --error=logs/debug/%j_%x.err




source ~/.bashrc
echo $me
cd $me/joint_training/cosmos-predict1
pwd


# -------------------------------
# 关键：用 srun 启动一条命令，在容器里跑你的脚本
# -------------------------------
srun --export=ALL -l \
     --container-image=$me/docker_images/imaginaire4_fsdp2.sqsh \
     --container-mounts=$me/joint_training/:/joint_training \
     bash /joint_training/cosmos-predict1/exp_scripts/env_entrance.sh $1 $2 $3 $4 $5 $6

exit_status=$?
echo "exit status code $exit_status"
