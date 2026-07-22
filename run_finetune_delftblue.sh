#!/bin/bash
#SBATCH --job-name=mswe-gnn-finetune
#SBATCH --account=education-ceg-msc-envm
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=5333M
#SBATCH --output=logs/%j_finetune.out
#SBATCH --error=logs/%j_finetune.err

# --- load modules (adjust versions with: module avail) ---
module load miniconda3

# --- activate environment ---
conda activate mswe-gnn

# --- move to repo root ---
cd $SLURM_SUBMIT_DIR

# wandb runs offline on compute nodes (no internet); sync manually after job with: wandb sync wandb/offline-run-*/
export WANDB_MODE=offline
export PYTHONPATH=$SLURM_SUBMIT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# NOTE: this script does NOT build datasets. The multisim/outflow-gc pkls must already
# exist under database/datasets/{train,test}/ before submitting (built via the
# run_convert_multisim*.py scripts, normally on hal8) — verify they're present on
# DelftBlue's filesystem (separate cluster/storage from hal8) before running.

# --- run fine-tuning ---
# Override via env vars when submitting: MSWE_CONFIG=... MSWE_OUTPUT=... sbatch run_finetune_delftblue.sh
CONFIG=${MSWE_CONFIG:-config_finetune_100m_velocity.yaml}
OUTPUT=${MSWE_OUTPUT:-results/finetuned_ahr.h5}
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

python -u finetune_ahr.py --config $CONFIG --output $OUTPUT \
    --checkpoint-dir lightning_logs/finetune_ahr/${SLURM_JOB_ID} \
    2>&1 | tee logs/${SLURM_JOB_ID}_finetune.log
