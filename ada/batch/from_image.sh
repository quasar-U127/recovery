#!/bin/bash
#SBATCH -A manan.sharma
#SBATCH --qos=normal
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --error="slurm-%j.err"
#SBATCH --output="slurm-%j.out"
#SBATCH --time=24:00:00
#SBATCH --mail-type=END


source ada/setup.sh
source ada/load.sh

data-manager --root $STGAN_DATA pull output/128/alteration


python main.py mobilenet_alteration \
    --root $PROJECT_DATA/mobilenet_alteration/all \
    --dataset $STGAN_DATA/output/128/alteration \
    train \
        --epochs 150 \
        --weight 25 1 25 \
        --save

# data-manager --root $PROJECT_DATA push .