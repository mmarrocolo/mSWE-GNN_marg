"""
Rebuild the warmstart pkl with the correct 94-node BC structure:
  A) 7 inflow face_bnd cells  — real WD from SFINCS (2-4 m peak)
  B) 87 outflow ghost cells   — WD=0 (free-drainage)
Both with type_BC=1 (fixed water level).
"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy, pickle
import numpy as np
import torch
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from database.convert_sfincs_to_pkl_marg import (
    load_single_data_object, get_target_points,
    parse_src_file, build_output_data,
)

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
WARMSTART_DIR = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations',
                             'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart')
SFINCS_MAP    = os.path.join(WARMSTART_DIR, 'sfincs_map.nc')
SFINCS_SRC    = os.path.join(WARMSTART_DIR, 'sfincs.src')
TEMPLATE_PKL  = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', 'template_100m.pkl')
OUT_ROOT      = os.path.join(PROJECT_ROOT, 'database', 'datasets')
DATASET_NAME  = 'ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart'

print('Loading template...')
template_data = load_single_data_object(TEMPLATE_PKL)
target_points = get_target_points(template_data)
num_targets   = target_points.shape[0]
print(f'  Target mesh faces: {num_targets}')

print('Opening SFINCS map...')
ds  = xr.open_dataset(SFINCS_MAP, decode_times=False)
msk = ds['msk'].values
zs  = ds['zs'].values   # [T, n, m]
zb  = ds['zb'].values   # [n, m]

if msk.ndim == 1:
    msk = msk.reshape(zs.shape[1], zs.shape[2])

x = ds.coords['x'].values
y = ds.coords['y'].values
if x.ndim == 1 and y.ndim == 1:
    if len(x) == msk.shape[1] and len(y) == msk.shape[0]:
        x, y = np.meshgrid(x, y, indexing='xy')
    elif x.shape[0] == msk.size:
        x = x.reshape(msk.shape)
        y = y.reshape(msk.shape)

active        = msk > 0
source_points = np.column_stack([x[active], y[active]])
print(f'  Active source cells: {source_points.shape[0]}')

time_var    = ds.coords.get('time', ds.coords.get('t', None))
map_times_s = time_var.values.astype(np.float64) if time_var is not None \
              else np.arange(zs.shape[0]) * 3600.0
print(f'  Time steps: {len(map_times_s)}  ({map_times_s[0]:.0f} – {map_times_s[-1]:.0f} s)')

# --- WD ---
print('Computing WD...')
zs_active = zs[:, active]
zb_active = zb[active]
zs_filled = np.where(np.isnan(zs_active), zb_active[None, :], zs_active)
WD_active  = np.maximum(zs_filled - zb_active[None, :], 0.0).astype(np.float32)
time_steps = WD_active.shape[0]
print(f'  Peak WD: {WD_active.max():.3f} m  |  ever-flooded: {(WD_active > 0.05).any(0).sum()}')

print('Interpolating WD to mesh...')
WD = np.zeros((num_targets, time_steps), dtype=np.float32)
for t in range(time_steps):
    iv = griddata(source_points, WD_active[t], target_points, method='linear')
    WD[:, t] = np.nan_to_num(iv, nan=0.0)
del WD_active, zs_active, zs_filled
print(f'  WD done: {WD.shape}  peak={WD.max():.3f} m')

# --- Velocities ---
ds_raw = xr.open_dataset(SFINCS_MAP, decode_times=False, mask_and_scale=False)
VX = np.zeros((num_targets, time_steps), dtype=np.float32)
VY = np.zeros((num_targets, time_steps), dtype=np.float32)

for var, arr_out in [('u', VX), ('v', VY)]:
    if var in ds_raw.data_vars:
        print(f'Interpolating velocity {var}...')
        raw = ds_raw[var].values.astype(np.float32)
        fv  = ds_raw[var].attrs.get('_FillValue', None)
        if fv is not None:
            raw[raw == float(fv)] = np.nan
        act = raw[:, active]
        del raw
        for t in range(time_steps):
            ok = np.isfinite(act[t])
            if ok.sum() > 3:
                iv = griddata(source_points[ok], act[t][ok], target_points, method='linear')
                arr_out[:, t] = np.nan_to_num(iv, nan=0.0)
        del act
        print(f'  {var} done.')
    else:
        print(f'  {var} not found — zeros.')
ds_raw.close()

# --- BC: 7 inflow (face_bnd) + 87 outflow (ghost) ---
print('\nBuilding BC...')
finest_mesh   = template_data.mesh.meshes[-1]
finest_offset = int(template_data.node_ptr[-2])

# A) 7 inflow face_bnd cells at discharge source locations
src_xy = parse_src_file(SFINCS_SRC)
face_bnd_local  = np.asarray(finest_mesh.face_bnd)
face_bnd_global = (face_bnd_local + finest_offset).astype(np.int64)
bnd_face_xy     = np.asarray(finest_mesh.face_xy)[face_bnd_local]

_, local_idx    = cKDTree(bnd_face_xy).query(src_xy)
inflow_global   = face_bnd_global[local_idx]   # [7]
inflow_face_xy  = bnd_face_xy[local_idx]        # [7, 2]
print(f'  7 inflow global indices: {inflow_global.tolist()}')

all_xy    = np.column_stack([x.reshape(-1), y.reshape(-1)])
_, sfincs_idx = cKDTree(all_xy).query(inflow_face_xy)   # [7] flat SFINCS indices
zs_flat   = zs.reshape(time_steps, -1)
zb_flat   = zb.reshape(-1)
wd_inflow = np.maximum(
    np.nan_to_num(zs_flat[:, sfincs_idx], nan=0.0) - zb_flat[sfincs_idx][None, :], 0.0
).astype(np.float32)   # [T, 7]
print(f'  Inflow peak WD: {wd_inflow.max(0).round(2)} m')

# B) 87 outflow ghost cells (msk==3), WD=0
ghost_global = template_data.node_BC.numpy().astype(np.int64)   # [87]
ghost_local  = ghost_global - finest_offset
ghost_xy     = np.asarray(finest_mesh.face_xy)[ghost_local]

msk3_flat = msk.reshape(-1) == 3
msk3_xy   = np.column_stack([x.reshape(-1)[msk3_flat], y.reshape(-1)[msk3_flat]])
_, g2m    = cKDTree(msk3_xy).query(ghost_xy)
msk3_wd   = np.maximum(
    np.nan_to_num(zs_flat[:, msk3_flat], nan=0.0) - zb_flat[msk3_flat][None, :], 0.0
).astype(np.float32)
wd_outlet = msk3_wd[:, g2m]   # [T, 87] all ~0
print(f'  Outlet ghost cells: {len(ghost_global)}  |  outlet WD peak: {wd_outlet.max():.3f} m')

del zs, zb, zs_flat, zb_flat, msk3_wd
ds.close()

# Combine
n_inflow  = len(inflow_global)   # 7
n_outflow = len(ghost_global)    # 87
n_bc      = n_inflow + n_outflow  # 94

node_bc_all = np.concatenate([inflow_global, ghost_global]).astype(np.int32)
bc_all      = np.zeros((n_bc, time_steps, 2), dtype=np.float32)
bc_all[:n_inflow, :, 1] = wd_inflow.T   # [7, T]
bc_all[n_inflow:, :, 1] = wd_outlet.T   # [87, T]

# Build data object
data_out = build_output_data(template_data, WD=WD, VX=VX, VY=VY, map_times_s=map_times_s)
data_out.node_BC        = torch.tensor(node_bc_all, dtype=torch.int32)
data_out.BC             = torch.FloatTensor(bc_all)
data_out.type_BC        = torch.tensor(1, dtype=torch.int32)
data_out.edge_BC_length = torch.ones(1, dtype=torch.float32)

print(f'\n=== BC summary ===')
print(f'  node_BC shape : {tuple(data_out.node_BC.shape)}  ({n_inflow} inflow + {n_outflow} outflow)')
print(f'  BC shape      : {tuple(data_out.BC.shape)}')
print(f'  type_BC       : {data_out.type_BC.item()} (1 = fixed WD)')
print(f'  inflow WD peak: {bc_all[:n_inflow,:,1].max():.3f} m')
print(f'  outlet WD peak: {bc_all[n_inflow:,:,1].max():.3f} m')

# Save
for split in ('train', 'test'):
    out_dir  = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{DATASET_NAME}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump([copy.deepcopy(data_out)], f)
    print(f'  Saved: {out_path}')

print(f'\nDone.  WD={tuple(data_out.WD.shape)}  VX={tuple(data_out.VX.shape)}  VY={tuple(data_out.VY.shape)}')
