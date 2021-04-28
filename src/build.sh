#!/bin/bash

#SBATCH -n 1
#SBATCH -t 1-00:00
##SBATCH -A ntallapr
#SBATCH -o inversecooking.%j.out
#SBATCH -e inversecooking.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ntallapr@asu.edu
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:GTX1080:2
#SBATCH --mem=40G

nvidia-smi
python3 build_vocab.py --recipe1m_path /scratch/ntallapr/inversecooking/data
python3 utils/ims2file.py --root /scratch/ntallapr/inversecooking/data

