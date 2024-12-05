#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=160gb
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hujie4529@gmail.com
#SBATCH --access=hk-project-p0022785


source ~/.bashrc
conda activate segmamba
cd /home/hk-project-cvhciass/tj3409/DeformableMamba

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29555}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="tools/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
