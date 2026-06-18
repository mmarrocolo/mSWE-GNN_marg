#!/bin/bash
#SBATCH --job-name=mswe-sweep
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=p-lcfgpu002
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=24:00:00
#SBATCH --output=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_sweep.out
#SBATCH --error=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_sweep.err
#SBATCH --mail-user=mmarrocolo@tudelft.nl
#SBATCH --mail-type=BEGIN,END,FAIL

# Usage:
#   1. On your LOCAL machine, initialise the sweep once:
#        wandb sweep sweep_config.yaml
#      This prints: "wandb: Created sweep with ID: <sweep_id>"
#
#   2. Submit N jobs on hal8 (one agent per job = one trial per job):
#        SWEEP_ID=<your-username>/mswe-gnn-ahr-sweep/0mnu7mxt
#        for i in $(seq 1 20); do sbatch --export=ALL,SWEEP_ID=$SWEEP_ID run_sweep_hal8.sh; done

module load miniconda3
conda activate mswe-gnn

cd $SLURM_SUBMIT_DIR
mkdir -p /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs

export PYTHONPATH=$SLURM_SUBMIT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

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
        2>&1 | tee /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/${SLURM_JOB_ID}_dataset.log
fi

# Run one sweep trial (--count 1 means this agent picks one config and exits cleanly)
wandb agent --count 1 $SWEEP_ID 2>&1 | tee /p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/${SLURM_JOB_ID}_sweep.log
