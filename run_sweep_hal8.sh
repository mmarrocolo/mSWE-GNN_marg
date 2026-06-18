#!/bin/bash
#SBATCH --job-name=mswe-sweep
#SBATCH --account=<your-account>          # TODO: fill in your hal8 account
#SBATCH --partition=<gpu-partition>       # TODO: fill in hal8 GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=5333M
#SBATCH --output=logs/%j_sweep.out
#SBATCH --error=logs/%j_sweep.err

# Usage:
#   1. On your LOCAL machine, initialise the sweep once:
#        wandb sweep sweep_config.yaml
#      This prints: "wandb: Created sweep with ID: <sweep_id>"
#
#   2. Export the sweep ID and submit N jobs (one agent per job, one run per agent):
#        export SWEEP_ID=<entity>/<project>/<sweep_id>
#        for i in $(seq 1 20); do sbatch --export=ALL,SWEEP_ID=$SWEEP_ID run_sweep_hal8.sh; done
#
#   If hal8 compute nodes have NO internet access, set WANDB_MODE=offline here and sync
#   manually after each job:  wandb sync wandb/offline-run-*/

module load miniconda3      # TODO: adjust to hal8 module name
conda activate mswe-gnn

cd $SLURM_SUBMIT_DIR
mkdir -p logs

export PYTHONPATH=$SLURM_SUBMIT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# Remove the next line if compute nodes have internet access:
# export WANDB_MODE=offline

# Build dataset PKL if missing
DATASET=database/datasets/train/ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart.pkl
SFINCS_DIR=database/raw_datasets_ahr/Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart

if [ ! -f "$DATASET" ]; then
    python database/convert_sfincs_to_pkl_marg.py \
        --sfincs-map   "$SFINCS_DIR/sfincs_map.nc" \
        --template-pkl database/datasets/train/template_100m.pkl \
        --dataset-name ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart \
        --out-root     database/datasets \
        --vx-var u --vy-var v \
        --src-file     "$SFINCS_DIR/sfincs.src" \
        --dis-file     "$SFINCS_DIR/sfincs.dis" \
        2>&1 | tee logs/${SLURM_JOB_ID}_dataset.log
fi

# Run one sweep trial (--count 1 means this agent picks exactly one config and exits)
wandb agent --count 1 $SWEEP_ID 2>&1 | tee logs/${SLURM_JOB_ID}_sweep.log
