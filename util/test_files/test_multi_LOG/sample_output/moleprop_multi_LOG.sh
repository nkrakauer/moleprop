#!/usr/bin/env bash

# Ron the short-list GPU queue
SBATCH --partition=sbel_cmg
SBATCH --account=skunkworks --qos=skunkworks_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -t 4-16:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH --error=~/Moleprop/summer19/test/model-%j.err
#SBATCH --output=~/Moleprop/summer19/test/model-%j.out

## Load CUDA into your environment
## load custimized CUDA and cudaToolkit

module load groupmods/cudnn/10.0
module load gcc/7.1.0 # provides libstdc++ compatible with everything in conda.

# activate virtual environment
conda activate deepchem
conda info --envs

#export HOME="/srv/home/nkrakauer/"
#export CUDA_HOME=/usr/local/cuda
#export PATH=$PATH:$CUDA_HOME/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$HOME/extras/CUPTI/lib64

# run the training scripts
python ~/Moleprop/summer19/util/test_multi_LOG/test_multi_LOG.py
