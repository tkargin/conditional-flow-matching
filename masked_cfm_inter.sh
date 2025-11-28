# Set up environment
module load miniforge

# Activate Python virtual environment
source ~/envs/torchcfm/bin/activate

# Run your application
python ~/projects/conditional-flow-matching/MaskedCFM/MaskedCFM_cluster.py