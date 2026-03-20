#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=baseline_experiment_1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=data/output/debug/experiment_1_debug_stdout.%j.txt
#SBATCH --error=data/output/debug/experiment_1_debug_stderr.%j.txt

# Source conda
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate quant_analysis

# Setup
cd ~/ondemand/dev/Quant_Analysis/
module unload cuda
module load cuda/12.8.0

# hopefully this fixes some thread errors
export OMP_NUM_THREADS=4
export ORT_NUM_THREADS=4
export TMPDIR=/tmp
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"

# checking the driver
nvidia-smi

# Run with proper redirection
python -m scripts.model_cluster_experiment