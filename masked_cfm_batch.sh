#!/bin/bash

# partition selection
#SBATCH -p mit_normal_gpu

# number of cpu cores
#SBATCH -c 16

# memory per core
# #SBATCH --mem=32G

# GPU cores
#SBATCH -G 4

# Set up environment
module load miniforge

# Activate Python virtual environment
source ~/envs/torchcfm/bin/activate

# Run your application
python ~/projects/conditional-flow-matching/MaskedCFM/MaskedCFM_cluster.py