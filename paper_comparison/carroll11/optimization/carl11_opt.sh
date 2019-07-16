#!/usr/bin/env bash

## Ron the short-list GPU queue
##SBATCH -p shortgpu

## modified by adminstrator
#SBATCH -p sbel_cmg --qos=skunkworks_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:1
#SBATCH -t 7-01:30 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH --error=/srv/home/xsun256//paper_comparison/carroll11/optimization/opt-carl10-%j.err
#SBATCH --output=/srv/home/xsun256//paper_comparison/carroll11/optimization/opt-carl10-%j.out

## Load CUDA
#module load usermods
module load cuda/10.0
module load groupmods/cudnn/10.0

## activate virtual environment
source activate deepchem

#pip install --upgrade pip
#pip uninstall -y tensorflow-gpu

## run the training scripts
python /srv/home/xsun256/paper_comparison/carroll11/optimization/opt.py

