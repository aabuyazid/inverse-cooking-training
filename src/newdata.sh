#!/bin/bash

#SBATCH -n 1
#SBATCH -t 0-03:00
##SBATCH -A ntallapr
#SBATCH -o demo.out
#SBATCH -e demo.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ntallapr@asu.edu
#SBATCH -p sulcgpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:GTX1080:2
#SBATCH --mem=60G

python3 newfile.py
