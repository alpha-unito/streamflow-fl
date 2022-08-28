#!/bin/bash

#SBATCH --partition=epito-12h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80

source /beegfs/home/icolonne/hybrid-federated-learning/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:/opt/pytorch-aarm64-a100/lib/python3.8/site-packages"
export PYTHONPATH="${PYTHONPATH}:/opt/vision-aarm64-a100/lib/python3.8/site-packages/torchvision-0.14.0a0+b4b246a-py3.8-linux-aarch64.egg"
export PYTHONPATH="${PYTHONPATH}:/opt/vision-aarm64-a100/lib/python3.8/site-packages/Pillow-9.2.0-py3.8-linux-aarch64.egg"

{{ streamflow_command }}