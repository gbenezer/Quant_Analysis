#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=baseline_experiment_1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=ondemand/dev/Quant_Analysis/data/output/debug/experiment_1_debug_stdout.%j.txt
#SBATCH --error=ondemand/dev/Quant_Analysis/data/output/debug/experiment_1_debug_stderr.%j.txt

# Source conda
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate quant_analysis

# checking on the torch and torchao versions installed just in case
python -c "import torch, torchao; print(torch.__version__, torchao.__version__)"

# Setup
cd ~/ondemand/dev/Quant_Analysis/
module unload cuda
module load cuda/12.8.0

# checking the driver
nvidia-smi

# Run with proper redirection
python -m scripts.model_cluster_experiment