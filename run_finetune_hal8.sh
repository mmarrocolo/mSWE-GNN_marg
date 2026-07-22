#!/bin/bash
#SBATCH --job-name=mswe-gnn-finetune
#SBATCH --output=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_finetune.out
#SBATCH --error=/p/11210554-dtc-hydrology-next/marrocol/mSWE-GNN_marg/logs/%j_finetune.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=24:00:00
#SBATCH --mail-user=mmarrocolo@tudelft.nl
#SBATCH --mail-type=BEGIN,END,FAIL

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

PYTHON=/u/marrocol/.conda/envs/mswe-gnn/bin/python

# --- build PKL dataset if not already present ---
# DATASET=database/datasets/train/ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart.pkl
# SFINCS_DIR=database/raw_datasets_ahr/Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart
# if [ ! -f "$DATASET" ]; then
#     $PYTHON database/convert_sfincs_to_pkl_marg.py \
#         --sfincs-map   "$SFINCS_DIR/sfincs_map.nc" \
#         --template-pkl database/datasets/train/template_100m.pkl \
#         --dataset-name ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart \
#         --out-root     database/datasets \
#         --vx-var u --vy-var v \
#         --src-file     "$SFINCS_DIR/sfincs.src" \
#         --dis-file     "$SFINCS_DIR/sfincs.dis" \
#         2>&1 | tee logs/${SLURM_JOB_ID}_dataset.log
# fi
DATASET=database/datasets/train/ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_3scales.pkl
TEMPLATE=database/datasets/train/template_100m_3scales.pkl
SFINCS_DIR=database/raw_datasets_ahr/Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon
SFINCS_DIR_WARMSTART=database/raw_datasets_ahr/Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart
if [ ! -f "$DATASET" ]; then
    if [ ! -f "$TEMPLATE" ]; then
        $PYTHON build_template_3scales.py \
            2>&1 | tee logs/${SLURM_JOB_ID}_template.log
    fi
    $PYTHON run_convert_warmstart_3scales.py \
        2>&1 | tee logs/${SLURM_JOB_ID}_dataset.log
fi

# --- run fine-tuning ---
# Override via env vars when submitting: MSWE_CONFIG=... MSWE_OUTPUT=... sbatch run_finetune_hal8.sh
# CONFIG=${MSWE_CONFIG:-config_finetune_100m_velocity.yaml}
# OUTPUT=${MSWE_OUTPUT:-results/finetuned_ahr.h5}
CONFIG=${MSWE_CONFIG:-config_best_sweep_3scales.yaml}
OUTPUT=${MSWE_OUTPUT:-results/best_sweep_3scales.h5}
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

srun $PYTHON -u finetune_ahr.py --config $CONFIG --output $OUTPUT \
    --checkpoint-dir lightning_logs/finetune_ahr/${SLURM_JOB_ID} \
    2>&1 | tee logs/${SLURM_JOB_ID}_finetune.log

echo "End time: $(date)"
