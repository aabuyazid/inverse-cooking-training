#!/bin/bash

#SBATCH -n 1
#SBATCH -t 1-00:00
#SBATCH -A ntallapr
#SBATCH -o inversecooking.%j.out
#SBATCH -e inversecooking.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ntallapr@asu.edu
#SBATCH -p rcgpu7
#SBATCH -q wildfire
#SBATCH --gres=gpu:K80:2
#SBATCH --mem=120G

python3 pretrained_model.py

