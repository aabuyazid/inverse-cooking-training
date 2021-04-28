#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3-00:00
##SBATCH -A ntallapr
#SBATCH -o inversecooking.%j.out
#SBATCH -e inversecooking.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ntallapr@asu.edu
#SBATCH -p rcgpu7
#SBATCH -q wildfire
#SBATCH --gres=gpu:K80:2
#SBATCH --mem=120G

nvidia-smi
python3 train.py --model_name im2ingr --batch_size 50 --num_workers 1 --finetune_after 0 --ingrs_only --load_jpeg\
    --es_metric iou_sample --loss_weight 0 1000.0 1.0 1.0 \
    --learning_rate 1e-4 --scale_learning_rate_cnn 1.0 \
    --save_dir /scratch/ntallapr/inversecooking/checkpoints --recipe1m_dir /scratch/ntallapr/inversecooking/data
#python3 train.py --model_name model --batch_size 20 --num_workers 1 --recipe_only --transfer_from im2ingr --load_jpeg\
    --save_dir /scratch/ntallapr/inversecooking/checkpoints --recipe1m_dir /scratch/ntallapr/inversecooking/data

python3 sample.py --model_name model --save_dir /scratch/ntallapr/inversecooking/checkpoints --recipe1m_dir /scratch/ntallapr/inversecooking/data --greedy --eval_split test
