"""
Warmstart dataset for Option B (full ghost cells: inflow + outflow).

BC structure (Option B):
  node_BC  = inflow ghost cells (msk==2 mirrors)  — prescribed WD from SFINCS msk==2 cells
  (outflow ghost cells msk==3 are NOT in node_BC — they evolve freely via message passing)

Template: template_100m_inflow_outflow_gc.pkl  (regenerated with Option B ghost cells)
Output:   ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_inflow_outflow_gc

Before running:
  1. Regenerate template:
       python database/create_mesh_template_marg.py \\
         --pol  <boundary.pol> --xyz <dem.xyz> \\
         --out  database/datasets/train/template_100m_inflow_outflow_gc.pkl \\
         --multiscale --num-scales 4 --sfincs <sfincs_map.nc>
     (Option B is the default now — add_ghost_cells_mesh auto-detects msk==2/3)
  2. Run this script on hal8.
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
    build_output_data, parse_src_file,
)

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
WARMSTART_DIR = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations',
                             'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart')
SFINCS_MAP    = os.path.join(WARMSTART_DIR, 'sfincs_map.nc')
SFINCS_SRC    = os.path.join(WARMSTART_DIR, 'sfincs.src')
TEMPLATE_PKL  = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                              'template_100m_inflow_outflow_gc.pkl')
OUT_ROOT      = os.path.join(PROJECT_ROOT, 'database', 'datasets')
DATASET_NAME  = 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_inflow_outflow_gc'

print('Loading template...')
template_data = load_single_data_object(TEMPLATE_PKL)
target_points = get_target_points(template_data)
num_targets   = target_points.shape[0]
print(f'  Target mesh faces: {num_targets}')

# node_BC == node_BC_inflow (inflow ghost cells only)
inflow_gc_global = template_data.node_BC.numpy().astype(np.int64)
n_inflow = len(inflow_gc_global)
print(f'  Inflow ghost cells (node_BC): {n_inflow}')

if hasattr(template_data, 'node_BC_outflow'):
    outflow_gc_global = template_data.node_BC_outflow.numpy().astype(np.int64)
    print(f'  Outflow ghost cells (not in node_BC): {len(outflow_gc_global)}')

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
active_flat   = np.flatnonzero(active.reshape(-1))
time_var      = ds.coords.get('time', ds.coords.get('t', None))
map_times_s   = time_var.values.astype(np.float64) if time_var is not None \
                else np.arange(zs.shape[0]) * 3600.0
time_steps    = zs.shape[0]
print(f'  Active source cells: {source_points.shape[0]}')
print(f'  Time steps: {time_steps}  ({map_times_s[0]:.0f} – {map_times_s[-1]:.0f} s)')

# --- WD for full mesh (interpolated) ---
print('Computing WD...')
zs_active = zs[:, active]
zb_active = zb[active]
zs_filled = np.where(np.isnan(zs_active), zb_active[None, :], zs_active)
WD_active  = np.maximum(zs_filled - zb_active[None, :], 0.0).astype(np.float32)
print(f'  Peak WD: {WD_active.max():.3f} m')

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

# --- BC: inflow cells — either msk==2 ghost cells (full Option B) or face_bnd fallback ---
print('\nBuilding BC for inflow cells...')
# After mesh_list[::-1] in template creation, SFINCS (finest) is at meshes[0].
# Templates built with the new code store finest_offset explicitly (= 0).
finest_mesh   = template_data.mesh.meshes[0]
finest_offset = int(template_data.finest_offset) if hasattr(template_data, 'finest_offset') \
                else int(template_data.node_ptr[-2])

zs_flat = zs.reshape(time_steps, -1)
zb_flat = zb.reshape(-1)
all_xy  = np.column_stack([x.reshape(-1), y.reshape(-1)])

if n_inflow > 0:
    # Full Option B: inflow ghost cells mirror msk==2 cells.
    # Get their WD from the warmstart map; fall back to nearest active cell if
    # the warmstart simulation has no msk==2 cells (uses sfincs.src instead).
    print(f'  Mode: inflow ghost cells (msk==2 mirrors)  n={n_inflow}')
    inflow_gc_local = (inflow_gc_global - finest_offset).astype(np.int64)
    inflow_gc_xy    = np.asarray(finest_mesh.face_xy)[inflow_gc_local]
    msk2_flat = msk.reshape(-1) == 2
    n_msk2    = msk2_flat.sum()
    print(f'  SFINCS msk==2 cells in warmstart map: {n_msk2}')
    if n_msk2 > 0:
        msk2_xy   = np.column_stack([all_xy[msk2_flat, 0], all_xy[msk2_flat, 1]])
        _, gc2ref = cKDTree(msk2_xy).query(inflow_gc_xy)
        ref_wd    = np.maximum(
            np.nan_to_num(zs_flat[:, msk2_flat], nan=0.0) - zb_flat[msk2_flat], 0.0
        ).astype(np.float32)
        wd_inflow = ref_wd[:, gc2ref]
        print(f'  Source: msk==2 cells → peak WD {wd_inflow.max():.3f} m')
    else:
        # Warmstart map has no msk==2 — map each ghost cell to nearest active cell.
        print('  Warmstart map has no msk==2 — mapping ghost cells to nearest active cell.')
        _, gc2sp    = cKDTree(source_points).query(inflow_gc_xy)
        gc2flat     = active_flat[gc2sp]
        wd_inflow   = np.maximum(
            np.nan_to_num(zs_flat[:, gc2flat], nan=0.0) - zb_flat[gc2flat], 0.0
        ).astype(np.float32)
        print(f'  Source: nearest active cell → peak WD {wd_inflow.max():.3f} m')
    node_bc_out = inflow_gc_global.astype(np.int32)

else:
    # Fallback: no msk==2 cells — use face_bnd cells nearest to sfincs.src sources
    # (standard for SFINCS models that use point-source discharge, not Neumann BC cells)
    print('  Mode: face_bnd cells near sfincs.src  (no msk==2 ghost cells in template)')
    src_xy          = parse_src_file(SFINCS_SRC)
    face_bnd_local  = np.asarray(finest_mesh.face_bnd)
    face_bnd_global = (face_bnd_local + finest_offset).astype(np.int64)
    bnd_face_xy     = np.asarray(finest_mesh.face_xy)[face_bnd_local]
    _, local_idx    = cKDTree(bnd_face_xy).query(src_xy)
    inflow_global   = face_bnd_global[local_idx]
    inflow_face_xy  = bnd_face_xy[local_idx]
    n_inflow        = len(inflow_global)
    print(f'  sfincs.src locations: {src_xy.shape[0]}  →  face_bnd inflow cells: {n_inflow}')
    print(f'  Global node indices: {inflow_global.tolist()}')
    _, sfincs_idx = cKDTree(all_xy).query(inflow_face_xy)
    wd_inflow     = np.maximum(
        np.nan_to_num(zs_flat[:, sfincs_idx], nan=0.0) - zb_flat[sfincs_idx], 0.0
    ).astype(np.float32)   # [T, n_inflow]
    print(f'  Inflow WD peak: {wd_inflow.max(0).round(2)} m')
    node_bc_out = inflow_global.astype(np.int32)

del zs, zb, zs_flat, zb_flat
ds.close()

# BC array: shape [N_in, T, 2]  (channel 0 = Q unused, channel 1 = WD)
bc_all = np.zeros((n_inflow, time_steps, 2), dtype=np.float32)
bc_all[:, :, 1] = wd_inflow.T   # [N_in, T]

# Build data object
data_out = build_output_data(template_data, WD=WD, VX=VX, VY=VY, map_times_s=map_times_s)
data_out.node_BC        = torch.tensor(node_bc_out, dtype=torch.int32)
data_out.BC             = torch.FloatTensor(bc_all)
data_out.type_BC        = torch.tensor(1, dtype=torch.int32)   # 1 = fixed WD
data_out.edge_BC_length = torch.ones(1, dtype=torch.float32)

# Carry over outflow ghost cell IDs (not in node_BC — they evolve freely)
if hasattr(template_data, 'node_BC_outflow'):
    data_out.node_BC_outflow = template_data.node_BC_outflow

print(f'\n=== BC summary ===')
print(f'  node_BC (inflow cells) : {tuple(data_out.node_BC.shape)}  (outflow ghost cells NOT in BC)')
print(f'  BC shape               : {tuple(data_out.BC.shape)}')
print(f'  type_BC                : {data_out.type_BC.item()} (1 = fixed WD)')
print(f'  Inflow WD peak         : {bc_all[:,:,1].max():.3f} m')

# Save
for split in ('train', 'test'):
    out_dir  = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{DATASET_NAME}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump([copy.deepcopy(data_out)], f)
    print(f'  Saved: {out_path}')

print(f'\nDone.  WD={tuple(data_out.WD.shape)}  VX={tuple(data_out.VX.shape)}  VY={tuple(data_out.VY.shape)}')
