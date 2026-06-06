"""Run inference with last.ckpt and print metrics."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib; matplotlib.use('Agg')

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from utils.load import read_config
from utils.miscellaneous import get_model, fix_dict_in_config
from utils.dataset import create_model_dataset, get_temporal_test_dataset_parameters, to_temporal_dataset
from utils.visualization import PlotRollout
from training.train import LightningTrainer

torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')

CHECKPOINT   = os.path.join(REPO_ROOT, 'lightning_logs', 'last.ckpt')
CONFIG       = 'config_finetune_100m_velocity.yaml'
DATASET_NAME = 'ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart'
OUTDIR       = os.path.join(REPO_ROOT, 'results', 'inference_last_ckpt')
os.makedirs(OUTDIR, exist_ok=True)

print(f'Checkpoint: {CHECKPOINT}')
print(f'Dataset:    {DATASET_NAME}')

# --- Load config and dataset ---
cfg = read_config(CONFIG)
cfg['dataset_parameters']['test_dataset_name']  = DATASET_NAME
cfg['dataset_parameters']['train_dataset_name'] = DATASET_NAME

wandb.init(mode='disabled', project='mswe-gnn', config=cfg)
fix_dict_in_config(wandb)
config = wandb.config
device = torch.device('cpu')

_, _, test_dataset, scalers = create_model_dataset(
    scalers=config.scalers, device=device,
    **config.dataset_parameters,
    **config.selected_node_features,
    **config.selected_edge_features
)
temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
    config, config.temporal_dataset_parameters
)
print(f'WD shape: {test_dataset[0].WD.shape}')

# --- Load model ---
temporal_test_dataset = to_temporal_dataset(
    test_dataset, rollout_steps=-1, **temporal_test_dataset_parameters
)
num_node_features = temporal_test_dataset[0].x.size(-1)
num_edge_features = temporal_test_dataset[0].edge_attr.size(-1)

model_parameters = dict(config.models)
model_type = model_parameters.pop('model_type')
if model_type == 'MSGNN':
    model_parameters['num_scales'] = test_dataset[0].mesh.num_meshes

_ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
_sd   = _ckpt['state_dict']
_hid  = _sd['model.edge_encoder.0.weight'].shape[0]
_K    = sum(1 for k in _sd if 'gnn_processor.0.filter_matrix.' in k and k.endswith('.weight')) - 1
if _hid != model_parameters.get('hid_features') or _K != model_parameters.get('K'):
    print(f'Checkpoint arch override: hid_features={_hid}, K={_K}')
    model_parameters['hid_features'] = _hid
    model_parameters['K'] = _K

model = get_model(model_type)(
    num_node_features=num_node_features,
    num_edge_features=num_edge_features,
    previous_t=temporal_test_dataset_parameters['previous_t'],
    device=device,
    **model_parameters
).to(device)

plmodule = LightningTrainer.load_from_checkpoint(
    CHECKPOINT, map_location=device,
    model=model,
    lr_info=config['lr_info'],
    trainer_options=config.trainer_options,
    temporal_test_dataset_parameters=temporal_test_dataset_parameters
)
model = plmodule.model.to(device)
model.eval()
print(f'Model loaded — epoch {_ckpt["epoch"]}')

# --- Rollout and metrics ---
plot_rollout = PlotRollout(
    model, test_dataset[0], scalers=scalers,
    warmup_steps=3,
    **temporal_test_dataset_parameters
)

rollout_loss = plot_rollout._get_rollout_loss(type_loss='RMSE')
loss_mean = rollout_loss.mean(0)
rmse_wd = loss_mean[0].item() if loss_mean.dim() > 0 else loss_mean.item()
csi_005 = plot_rollout._get_CSI(water_threshold=0.05).nanmean().item()
csi_03  = plot_rollout._get_CSI(water_threshold=0.30).nanmean().item()

print('\n=== RESULTS ===')
print(f'RMSE WD:      {rmse_wd:.4f} m')
print(f'CSI @ 0.05 m: {csi_005:.4f}')
print(f'CSI @ 0.30 m: {csi_03:.4f}')

# --- CSI over time plot ---
fig, ax = plt.subplots(figsize=(10, 4))
for thr in [0.05, 0.30]:
    csi_t = plot_rollout._get_CSI(water_threshold=thr).detach().cpu().numpy()
    if csi_t.ndim > 1:
        csi_t = np.nanmean(csi_t, axis=0)
    ax.plot(csi_t, label=f'CSI @ {thr:.2f} m')
ax.set_xlabel('Time step [h]'); ax.set_ylabel('CSI')
ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend()
ax.set_title('CSI over time — last.ckpt (epoch 521)')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'csi_over_time.png'), dpi=150)
print(f'Saved: results/inference_last_ckpt/csi_over_time.png')

# --- Snapshot at peak time ---
n_steps = plot_rollout.real_rollout.shape[-1]
for t in [0, 24, 48, 72, n_steps-1]:
    fig = plot_rollout.explore_rollout(scale=0, time_step=t)
    fname = os.path.join(OUTDIR, f'snapshot_t{t:03d}.png')
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: results/inference_last_ckpt/snapshot_t{t:03d}.png')

print('\nDone.')
