#!/bin/bash
#SBATCH --job-name=mswe-gnn-timing
#SBATCH --output=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_timing.out
#SBATCH --error=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_timing.err
#SBATCH --partition=4vcpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=01:00:00
#SBATCH --mail-user=mmarrocolo@tudelft.nl
#SBATCH --mail-type=BEGIN,END,FAIL

# Dedicated 4-core CPU-only allocation, matched to the SFINCS runs being compared
# against: every warmstart_q0XX sfincs.log reports "Using 4 of 4 available threads".
# Do NOT run this timing notebook on the login/submit node directly -- CPU time
# there is shared and unallocated, which produced meaningless numbers (GNN times
# that decayed from 310s to 58s across identical-architecture events).

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

mkdir -p /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs

source /opt/apps/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh
conda activate mswe-gnn
export PATH="/u/marrocol/.conda/envs/mswe-gnn/bin:$PATH"

cd /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg
export PYTHONPATH=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg

srun jupyter nbconvert --to notebook --execute --inplace \
    utils/measure_ahr_inference_timing.ipynb \
    2>&1 | tee logs/${SLURM_JOB_ID}_timing.log

echo "End time: $(date)"
