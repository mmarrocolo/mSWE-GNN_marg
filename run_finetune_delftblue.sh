#!/bin/bash
#SBATCH --job-name=mswe-gnn-finetune
#SBATCH --account=education-ceg-msc-envm
#SBATCH --partition=gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=5333M
#SBATCH --output=logs/%j_finetune.out
#SBATCH --error=logs/%j_finetune.err

# --- load modules (adjust versions with: module avail) ---
module load miniconda3

# --- activate environment ---
conda activate mswe-gnn

# --- move to repo root ---
cd $SLURM_SUBMIT_DIR

# --- disable wandb online sync (no internet on compute nodes) ---
export WANDB_MODE=offline

# --- run fine-tuning ---
python finetune_ahr.py --config config_finetune_100m_velocity.yaml 2>&1 | tee logs/${SLURM_JOB_ID}_finetune.log
