#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=baseline_experiment_1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

# Source conda
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate quant_analysis

# Setup
cd ~/ondemand/dev/Quant_Analysis/
module unload cuda
module load cuda/12.8.0

# Run with proper redirection
python -m scripts.model_cluster_experiment > "data/output/debug/experiment_1_debug_output.txt" 2>&1