"""Convert the warmstart simulations (1.0x + the 4 scaled hydrographs) onto the
OUTFLOW ghost-cell template, for the absorbing-outflow experiment.
(No inflow ghost cells exist — the model has no msk==2 boundary; inflow is the
discharge at the 7 .src points — hence the plain "outflow" naming.)

PREREQUISITE — build the outflow template FIRST (bidirectional ghost edges,
database/graph_creation.py fix, Jul 2026):

    python build_template_outflow_gc.py

Then:  python run_convert_multisim_outflow.py

Outputs (train/ and test/ for per-sim, train/ for the merged):
  ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_outflow_gc                  (1.0x, val/test)
  ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_{q050,q075,q125,q150}_outflow_gc  (per factor)
  ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_outflow_gc_multisim         (4 events, train)

Same conversion logic as run_convert_warmstart_inflow_outflow_gc.py, looped.
Every dataset carries node_BC_outflow -> training/rollout apply the absorbing
WD=0 clamp at the ghost cells automatically (training/train.py).
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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT     = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations')
TEMPLATE_PKL = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                            'template_100m_outflow_gc.pkl')
OUT_ROOT     = os.path.join(PROJECT_ROOT, 'database', 'datasets')

BASE_FOLDER = 'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart'
# tag -> (sim folder, output dataset name)
SIMS = {
    '1x':   (BASE_FOLDER,           'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_outflow_gc'),
    'q050': (BASE_FOLDER + '_q050', 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_q050_outflow_gc'),
    'q075': (BASE_FOLDER + '_q075', 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_q075_outflow_gc'),
    'q125': (BASE_FOLDER + '_q125', 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_q125_outflow_gc'),
    'q150': (BASE_FOLDER + '_q150', 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_q150_outflow_gc'),
}
MERGED_NAME = 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_outflow_gc_multisim'

assert os.path.exists(TEMPLATE_PKL), \
    f'Template not found: {TEMPLATE_PKL} — run build_template_outflow_gc.py first'

print('Loading gc template...')
template_data = load_single_data_object(TEMPLATE_PKL)
target_points = get_target_points(template_data)
num_targets   = target_points.shape[0]
assert hasattr(template_data, 'node_BC_outflow'), 'template has no node_BC_outflow — wrong template?'
print(f'  Target mesh faces: {num_targets}')
print(f'  Outflow ghost cells: {len(template_data.node_BC_outflow)}')


def convert_one(sim_dir):
    """SFINCS map -> (WD, VX, VY, times, src, dis) -> Data object on the gc template."""
    sfincs_map = os.path.join(sim_dir, 'sfincs_map.nc')
    ds  = xr.open_dataset(sfincs_map, decode_times=False)
    msk = ds['msk'].values
    zs  = ds['zs'].values
    zb  = ds['zb'].values
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

    # WD (NaN = dry -> fill with bed level so it enters as WD=0)
    zs_active = zs[:, active]
    zb_active = zb[active]
    zs_filled = np.where(np.isnan(zs_active), zb_active[None, :], zs_active)
    WD_active = np.maximum(zs_filled - zb_active[None, :], 0.0).astype(np.float32)
    WD = np.zeros((num_targets, time_steps), dtype=np.float32)
    for t in range(time_steps):
        iv = griddata(source_points, WD_active[t], target_points, method='linear')
        WD[:, t] = np.nan_to_num(iv, nan=0.0)
    del WD_active, zs_active, zs_filled

    # Velocities
    ds_raw = xr.open_dataset(sfincs_map, decode_times=False, mask_and_scale=False)
    VX = np.zeros((num_targets, time_steps), dtype=np.float32)
    VY = np.zeros((num_targets, time_steps), dtype=np.float32)
    for var, arr_out in [('u', VX), ('v', VY)]:
        if var in ds_raw.data_vars:
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
    ds_raw.close()

    # Discharge BC (type_BC=2)
    src_xy = parse_src_file(os.path.join(sim_dir, 'sfincs.src'))
    dis_times_s, discharge = parse_dis_file(os.path.join(sim_dir, 'sfincs.dis'))
    del zs, zb
    ds.close()

    data_out = build_output_data(template_data, WD=WD, VX=VX, VY=VY,
                                 map_times_s=map_times_s,
                                 src_xy=src_xy, dis_times_s=dis_times_s, discharge=discharge)
    data_out.node_BC_outflow = template_data.node_BC_outflow
    return data_out


# --- convert every simulation ---
per_sim = {}
for tag, (folder, dataset_name) in SIMS.items():
    sim_dir = os.path.join(SIM_ROOT, folder)
    for fname in ['sfincs_map.nc', 'sfincs.src', 'sfincs.dis']:
        assert os.path.exists(os.path.join(sim_dir, fname)), f'{tag}: missing {fname} in {sim_dir}'

    out_train = os.path.join(OUT_ROOT, 'train', dataset_name + '.pkl')
    if os.path.exists(out_train):
        print(f'\n=== {tag}: already converted, loading ({dataset_name}) ===')
        with open(out_train, 'rb') as f:
            per_sim[tag] = pickle.load(f)[0]
        continue

    print(f'\n=== converting {tag} ({folder}) ===')
    data_out = convert_one(sim_dir)
    per_sim[tag] = data_out
    q_peak = float(data_out.BC[:, :, 1].max())
    print(f'  WD={tuple(data_out.WD.shape)}  Q peak={q_peak:.1f} m3/s  '
          f'type_BC={int(data_out.type_BC)}  outflow gc={len(data_out.node_BC_outflow)}')
    for split in ('train', 'test'):
        os.makedirs(os.path.join(OUT_ROOT, split), exist_ok=True)
        with open(os.path.join(OUT_ROOT, split, dataset_name + '.pkl'), 'wb') as f:
            pickle.dump([copy.deepcopy(data_out)], f)
    print(f'  saved train+test: {dataset_name}.pkl')

# --- merge the 4 scaled events (1.0x stays OUT: it is the validation/test dataset) ---
q_ref = float(per_sim['1x'].BC[:, :, 1].max())
print(f'\n=== merging (1.0x ref Q peak = {q_ref:.1f} m3/s) ===')
merged = []
for tag in ['q050', 'q075', 'q125', 'q150']:
    d = per_sim[tag]
    q = float(d.BC[:, :, 1].max())
    same_bc = d.node_BC.tolist() == per_sim['1x'].node_BC.tolist()
    print(f'  {tag}: Q peak = {q:8.1f} (ratio {q/q_ref:.3f})   WD peak = {float(d.WD.max()):.3f} m   '
          f'node_BC identical: {same_bc}')
    assert same_bc, f'{tag}: node_BC differs from the 1.0x event!'
    merged.append(d)

out_path = os.path.join(OUT_ROOT, 'train', MERGED_NAME + '.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(merged, f)
print(f'\nSaved {len(merged)} events -> {out_path}')
print('Sanity: Q-peak ratios must read ~0.5 / 0.75 / 1.25 / 1.5.')
print('Train with config_best_sweep_multisim_msloss_outflow.yaml.')
