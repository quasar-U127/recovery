#!/bin/bash
#SBATCH -A manan.sharma
#SBATCH --qos=normal
#SBATCH -n 10
#SBATCH --time=12:00:00
#SBATCH --mail-type=END


source ada/setup.sh
source ada/load.sh

data-manager --root $STGAN_DATA pull output/128/alteration

atts="[Bald] [Bangs] [Black_Hair] [Blond_Hair] [Brown_Hair] [Bushy_Eyebrows] [Eyeglasses] [Male] [Mouth_Slightly_Open] [Mustache] [No_Beard] [Pale_Skin] [Young]"

for att in $atts
do
    python main.py alteration --root $PROJECT_DATA/alteration/"$att" --dataset $STGAN_DATA/output/128/alteration/"$att" train
done

data-manager --root $PROJECT_DATA push .