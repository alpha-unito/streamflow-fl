#!/bin/bash

#SBATCH --account=IscrC_DFL
#SBATCH --partition=m100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load anaconda/2020.11
module load cuda/11.0
module load profile/deeplrn
module load cineca-ai/2.2.0

{{ streamflow_command }}