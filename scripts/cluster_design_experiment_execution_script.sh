#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --job-name=design_space_exploration_experiment_1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=data/output/debug/design_space_exploration_experiment_1_debug_stdout.%j.txt
#SBATCH --error=data/output/debug/design_space_exploration_experiment_1_debug_stderr.%j.txt

# Source conda
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate quant_analysis

# Setup
cd ~/ondemand/dev/Quant_Analysis/
module unload cuda
module load cuda/12.1.1

# ORT thread count: prevents pthread_setaffinity_np errors from ORT
# spawning more threads than SLURM-allocated CPUs (cpus-per-task=5)
export OMP_NUM_THREADS=4
export ORT_NUM_THREADS=4

# Redirect temp dirs away from SLURM scratch to prevent OSError [Errno 16]
# on cleanup when multiprocessing workers exit
export TMPDIR=/tmp
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"

# checking the driver
nvidia-smi

# Run with proper redirection
python -m scripts.cluster_design_space_sampling_experiment