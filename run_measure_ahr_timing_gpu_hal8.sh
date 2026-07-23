#!/bin/bash
#SBATCH --job-name=mswe-gnn-timing-gpu
#SBATCH --output=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_timing_gpu.out
#SBATCH --error=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_timing_gpu.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=01:00:00
#SBATCH --mail-user=mmarrocolo@tudelft.nl
#SBATCH --mail-type=BEGIN,END,FAIL

# GPU counterpart of run_measure_ahr_timing_hal8.sh: runs the GNN on GPU instead
# of 4 CPU cores, to measure the speed-up regime the mSWE-GNN paper's claim is
# actually about (GPU-parallel DL inference vs a CPU numerical solver).

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start time: $(date)"

mkdir -p /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs

source /opt/apps/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh
conda activate mswe-gnn
export PATH="/u/marrocol/.conda/envs/mswe-gnn/bin:$PATH"

cd /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg
export PYTHONPATH=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg

srun jupyter nbconvert --to notebook --execute --inplace \
    utils/measure_ahr_inference_timing_gpu.ipynb \
    2>&1 | tee logs/${SLURM_JOB_ID}_timing_gpu.log

echo "End time: $(date)"
