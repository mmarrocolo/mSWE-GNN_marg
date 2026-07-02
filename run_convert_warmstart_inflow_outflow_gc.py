"""
Warmstart dataset with outflow ghost cells and discharge (Q) inflow BC.

BC structure:
  node_BC = 7 boundary faces nearest the sfincs.src injection points
  BC[:,:,1] = Q [m^3/s] from sfincs.dis interpolated to map time steps
  type_BC = 2 (discharge -> VX slot in apply_boundary_condition)
  Outflow ghost cells (msk==3, node_BC_outflow) are NOT in node_BC — they evolve
  freely via message passing.

NOTE: this SFINCS model has no msk==2 boundary cells (inflow comes from .src point
sources), so no inflow ghost cells exist. An earlier version of this script fell
back to prescribing ground-truth WD at the source cells (type_BC=1) — that leaked
the SFINCS solution into the input and never exposed the model to the hydrograph.

Template: template_100m_inflow_outflow_gc.pkl
Output:   ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_inflow_outflow_gc
"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy, pickle
import numpy as np
import torch
import xarray as xr
from scipy.interpolate import griddata

from database.convert_sfincs_to_pkl_marg import (
    load_single_data_object, get_target_points,
    build_output_data, parse_src_file, parse_dis_file,
)

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
WARMSTART_DIR = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations',
                             'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart')
SFINCS_MAP    = os.path.join(WARMSTART_DIR, 'sfincs_map.nc')
SFINCS_SRC    = os.path.join(WARMSTART_DIR, 'sfincs.src')
SFINCS_DIS    = os.path.join(WARMSTART_DIR, 'sfincs.dis')
TEMPLATE_PKL  = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                              'template_100m_inflow_outflow_gc.pkl')
OUT_ROOT      = os.path.join(PROJECT_ROOT, 'database', 'datasets')
DATASET_NAME  = 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_inflow_outflow_gc'

print('Loading template...')
template_data = load_single_data_object(TEMPLATE_PKL)
target_points = get_target_points(template_data)
num_targets   = target_points.shape[0]
print(f'  Target mesh faces: {num_targets}')

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

# --- BC: discharge Q at the 7 sfincs.src injection points (type_BC=2) ---
print('\nBuilding discharge BC from sfincs.src / sfincs.dis...')
src_xy = parse_src_file(SFINCS_SRC)
dis_times_s, discharge = parse_dis_file(SFINCS_DIS)
print(f'  {len(src_xy)} source points  |  Q peak: {discharge.max():.1f} m^3/s')

del zs, zb
ds.close()

# build_output_data maps src -> face_bnd cells, interpolates Q to map times,
# sets type_BC=2 and edge_BC_length
data_out = build_output_data(template_data, WD=WD, VX=VX, VY=VY,
                             map_times_s=map_times_s,
                             src_xy=src_xy, dis_times_s=dis_times_s, discharge=discharge)

# Carry over outflow ghost cell IDs (not in node_BC — they evolve freely)
if hasattr(template_data, 'node_BC_outflow'):
    data_out.node_BC_outflow = template_data.node_BC_outflow

print(f'\n=== BC summary ===')
print(f'  node_BC (inflow cells) : {tuple(data_out.node_BC.shape)}  (outflow ghost cells NOT in BC)')
print(f'  node_BC indices        : {data_out.node_BC.tolist()}')
print(f'  BC shape               : {tuple(data_out.BC.shape)}')
print(f'  type_BC                : {data_out.type_BC.item()} (2 = discharge -> VX slot)')
print(f'  Q peak                 : {data_out.BC[:,:,1].max().item():.1f} m^3/s')

# Save
for split in ('train', 'test'):
    out_dir  = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{DATASET_NAME}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump([copy.deepcopy(data_out)], f)
    print(f'  Saved: {out_path}')

print(f'\nDone.  WD={tuple(data_out.WD.shape)}  VX={tuple(data_out.VX.shape)}  VY={tuple(data_out.VY.shape)}')
