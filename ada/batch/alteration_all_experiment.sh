#!/bin/bash
#SBATCH -A manan.sharma
#SBATCH --qos=normal
#SBATCH -n 10
#SBATCH --time=12:00:00
#SBATCH --mail-type=END


source ada/setup.sh
source ada/load.sh

data-manager --root $STGAN_DATA pull output/128/alteration


python main.py alteration --root $PROJECT_DATA/alteration --dataset $STGAN_DATA/output/128/alteration/ train

data-manager --root $PROJECT_DATA push .