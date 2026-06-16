"""
Rebuild template_marg.pkl with the 100 m SFINCS mesh as the finest level,
then recreate ahr_river_v03_marg_additionalsrc_cutpolygon.pkl by
interpolating the 10 m simulation WD onto the new mesh.

Run from the repo root:
    conda activate mswe-gnn
    python database/make_ahr_sfincs_template.py
"""

import copy
import os
import pickle
import sys

import numpy as np
import torch
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# ── make graph_creation importable ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from create_mesh_template_marg import create_mesh_template_from_pol

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW          = os.path.join(SCRIPT_DIR, 'raw_datasets_ahr')
TRAIN_DIR    = os.path.join(SCRIPT_DIR, 'datasets', 'train')
TEST_DIR     = os.path.join(SCRIPT_DIR, 'datasets', 'test')

POL_FILE     = os.path.join(RAW, 'Geometry',    'Polygon_1.pol')
DEM_FILE     = os.path.join(RAW, 'DEM',         'DEM_1.xyz')
SFINCS_100m  = os.path.join(RAW, 'Simulations', 'output_1_map.nc')
SFINCS_10m   = os.path.join(RAW, 'Simulations',
               'ahr_river_v03_Marg_additionalsrc_cutpolygon', 'sfincs_map.nc')

TEMPLATE     = os.path.join(TRAIN_DIR, 'template_marg.pkl')
DATASET_NAME = 'ahr_river_v03_marg_additionalsrc_cutpolygon'
EXISTING_PKL = os.path.join(TRAIN_DIR, f'{DATASET_NAME}.pkl')

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Rebuild template with 100 m SFINCS mesh as finest level
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Step 1: Building template with SFINCS 100 m mesh ===')
create_mesh_template_from_pol(
    pol_path=POL_FILE,
    xyz_path=DEM_FILE,
    output_pkl_path=TEMPLATE,
    with_multiscale=True,
    number_of_multiscales=4,
    sfincs_map_nc=SFINCS_100m,
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Rebuild additionalsrc dataset
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n=== Step 2: Rebuilding {DATASET_NAME} ===')

# Load new template
with open(TEMPLATE, 'rb') as f:
    template_data = pickle.load(f)[0]
target_points = np.asarray(template_data.mesh.face_xy)  # (N_new, 2)
print(f'  New mesh faces: {target_points.shape[0]}')

# Load existing pkl to preserve BC geographic locations and discharge time series
if not os.path.exists(EXISTING_PKL):
    raise FileNotFoundError(
        f'Existing dataset not found: {EXISTING_PKL}\n'
        'Run the original dataset creation first before rebuilding with the new template.'
    )
with open(EXISTING_PKL, 'rb') as f:
    existing_data = pickle.load(f)[0]
print(f'  BC preserved from existing pkl: {tuple(existing_data.BC.shape)}')

# Map old BC face locations → nearest faces in the new mesh
old_bc_ids  = existing_data.node_BC.numpy().astype(int)
old_face_xy = np.asarray(existing_data.mesh.face_xy)
bc_geo_xy   = old_face_xy[old_bc_ids]        # geographic coords of BC nodes

tree = cKDTree(target_points)
_, new_bc_ids = tree.query(bc_geo_xy)        # nearest face in new mesh
print(f'  BC nodes remapped: {old_bc_ids.tolist()} → {new_bc_ids.tolist()}')

# ── Interpolate WD from the 10 m SFINCS output onto the new mesh ──────────────
# Stream one time step at a time to avoid loading the full (T, n, m) array (~4 GB).
ds  = xr.open_dataset(SFINCS_10m, decode_times=False)
x   = ds.coords['x'].values          # (n_rows, n_cols) 10 m cell centres
y   = ds.coords['y'].values
zb  = ds['zb'].values.astype(np.float32)   # (n_rows, n_cols)  — small, load once
T   = ds.dims['time']

src_pts = np.column_stack([x.reshape(-1), y.reshape(-1)])
WD      = np.zeros((target_points.shape[0], T), dtype=np.float32)

print(f'  Interpolating {T} time steps ({src_pts.shape[0]} source pts → {target_points.shape[0]} target pts) ...')
for t in range(T):
    zs_t = ds['zs'].isel(time=t).values.astype(np.float32)   # (n_rows, n_cols)
    wd_t = np.maximum(zs_t - zb, 0.0).reshape(-1)
    valid = np.isfinite(wd_t)
    if not valid.any():
        continue
    interp = griddata(src_pts[valid], wd_t[valid], target_points, method='linear')
    nan_mask = ~np.isfinite(interp)
    if nan_mask.any():
        interp[nan_mask] = griddata(
            src_pts[valid], wd_t[valid], target_points[nan_mask], method='nearest'
        )
    WD[:, t] = np.nan_to_num(interp, nan=0.0)
    if (t + 1) % 20 == 0 or t + 1 == T:
        print(f'    t={t + 1}/{T}  WD_max={WD[:, t].max():.3f} m')

# ── Assemble the new Data object ───────────────────────────────────────────────
data = copy.deepcopy(template_data)
data.WD = torch.FloatTensor(WD)
data.VX = torch.zeros_like(data.WD)
data.VY = torch.zeros_like(data.WD)

data.node_BC       = torch.tensor(new_bc_ids, dtype=torch.int32)
data.BC            = existing_data.BC.clone()           # discharge time series unchanged
data.type_BC       = existing_data.type_BC.clone()
data.edge_BC_length = torch.ones(len(new_bc_ids), 1, dtype=torch.float32)

# Carry over temporal resolution if present
if hasattr(existing_data, 'temporal_res'):
    data.temporal_res = existing_data.temporal_res
else:
    data.temporal_res = 60  # minutes, default for this dataset

# ── Save train and test copies ─────────────────────────────────────────────────
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)

train_path = os.path.join(TRAIN_DIR, f'{DATASET_NAME}.pkl')
test_path  = os.path.join(TEST_DIR,  f'{DATASET_NAME}.pkl')

with open(train_path, 'wb') as f:
    pickle.dump([data], f)
with open(test_path, 'wb') as f:
    pickle.dump([copy.deepcopy(data)], f)

print(f'\n=== Done ===')
print(f'  Template:  {TEMPLATE}')
print(f'  Train pkl: {train_path}')
print(f'  Test pkl:  {test_path}')
print(f'  WD shape:  {tuple(data.WD.shape)}')
print(f'  BC shape:  {tuple(data.BC.shape)}')
print(f'  node_BC:   {data.node_BC.tolist()}')
