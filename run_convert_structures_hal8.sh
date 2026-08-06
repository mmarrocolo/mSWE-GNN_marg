#!/bin/bash
#SBATCH --job-name=mswe-gnn-convert-structures
#SBATCH --output=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_convert_structures.out
#SBATCH --error=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_convert_structures.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=04:00:00
#SBATCH --mail-user=mmarrocolo@tudelft.nl
#SBATCH --mail-type=BEGIN,END,FAIL

# No --gres=gpu here on purpose - this is pure CPU/numpy/scipy data conversion work,
# no model forward passes. Check `sinfo` if "normal" isn't a valid partition name on
# this cluster and swap it for whatever the CPU-only partition is called.

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

mkdir -p /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs

source /opt/apps/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh
conda activate mswe-gnn
export PATH="/u/marrocol/.conda/envs/mswe-gnn/bin:$PATH"

cd /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg
export PYTHONPATH=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg

PYTHON=/u/marrocol/.conda/envs/mswe-gnn/bin/python

$PYTHON -u run_convert_structures.py 2>&1 | tee logs/${SLURM_JOB_ID}_convert_structures.log

echo "End time: $(date)"
