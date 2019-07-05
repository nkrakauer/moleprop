#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH --partition=sbel_cmg
#SBATCH --account=skunkworks --qos=skunkworks_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -t 4-16:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH --error=/srv/home/xsun256/paper_comparison/pan07/test-pan07-%j.err
#SBATCH --output=/srv/home/xsun256/paper_comparison/pan07/test-pan07-%j.out

## Load CUDA into your environment
## load custimized CUDA and cudaToolkit

#module load usermods
module load cuda/10.0
module load groupmods/cudnn/10.0

# activate virtual environment
conda activate deepchem
conda info --envs

#export HOME="/srv/home/xsun256/"
#export CUDA_HOME=/usr/local/cuda
#export PATH=$PATH:$CUDA_HOME/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$HOME/extras/CUPTI/lib64

python /srv/home/xsun256/paper_comparison/pan07/test_multi_LOG.py

