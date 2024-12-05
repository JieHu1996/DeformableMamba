#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=160gb
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hujie4529@gmail.com

source ~/.bashrc
conda activate vssm
cd /home/ka/ka_iar/ka_ba9856/mmsegmentation/mmseg/models/utils/dcnv3
bash make.sh