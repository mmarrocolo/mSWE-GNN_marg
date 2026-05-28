"""
Patch the warmstart pkl to use discharge BCs (type_BC=2), matching the
non-warmstart training pkl format exactly.
Uses build_output_data() with src/dis args — the same path as training data.
"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy, pickle
import numpy as np
import torch
import xarray as xr
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
WARMSTART_DIR = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations',
                             'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart')
SFINCS_MAP    = os.path.join(WARMSTART_DIR, 'sfincs_map.nc')
SFINCS_SRC    = os.path.join(WARMSTART_DIR, 'sfincs.src')
SFINCS_DIS    = os.path.join(WARMSTART_DIR, 'sfincs.dis')
TEMPLATE_PKL  = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', 'template_100m.pkl')
OUT_ROOT      = os.path.join(PROJECT_ROOT, 'database', 'datasets')
DATASET_NAME  = 'ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart'


def parse_src(src_path):
    pts = []
    with open(src_path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    pts.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
    return np.array(pts, dtype=np.float64)


def parse_dis(dis_path):
    data = np.loadtxt(dis_path)
    return data[:, 0], data[:, 1:].astype(np.float32)


# Step 1 — load template (mesh topology, existing WD/VX/VY)
print('Loading template pkl ...', flush=True)
with open(TEMPLATE_PKL, 'rb') as f:
    template_data = pickle.load(f)[0]
finest_mesh   = template_data.mesh.meshes[-1]
finest_offset = int(template_data.node_ptr[-2])
print(f'  finest_offset={finest_offset}', flush=True)

# Step 2 — find face_bnd nodes nearest to src points
print('Mapping src -> face_bnd ...', flush=True)
src_xy = parse_src(SFINCS_SRC)
face_bnd_local  = np.asarray(finest_mesh.face_bnd)
face_bnd_global = (face_bnd_local + finest_offset).astype(np.int64)
bnd_face_xy     = np.asarray(finest_mesh.face_xy)[face_bnd_local]

_, local_idx   = cKDTree(bnd_face_xy).query(src_xy)
node_bc        = face_bnd_global[local_idx].astype(np.int32)   # [7]
print(f'  node_BC: {node_bc.tolist()}', flush=True)

# Step 3 — get map time vector
print('Reading map times ...', flush=True)
ds = xr.open_dataset(SFINCS_MAP, decode_times=False)
time_var = ds.coords.get('time', ds.coords.get('t', None))
zs_shape = ds['zs'].values.shape   # just read shape, not values
ds.close()
n_time = zs_shape[0]
if time_var is not None:
    map_times_s = ds.coords.get('time', ds.coords.get('t', None))
    # Re-open to get actual values
    ds2 = xr.open_dataset(SFINCS_MAP, decode_times=False)
    t_key = 'time' if 'time' in ds2.coords else 't'
    map_times_s = ds2.coords[t_key].values.astype(np.float64)
    ds2.close()
else:
    map_times_s = np.arange(n_time) * 3600.0
print(f'  {len(map_times_s)} steps  {map_times_s[0]:.0f}-{map_times_s[-1]:.0f} s', flush=True)

# Step 4 — interpolate discharge to map times
print('Interpolating discharge ...', flush=True)
dis_t, discharge = parse_dis(SFINCS_DIS)
n_bc = len(node_bc)
time_steps = len(map_times_s)
bc_tensor = np.zeros((n_bc, time_steps, 2), dtype=np.float32)
for i in range(n_bc):
    f = interp1d(dis_t, discharge[:, i], kind='linear', bounds_error=False,
                 fill_value=(discharge[0, i], discharge[-1, i]))
    bc_tensor[i, :, 1] = f(map_times_s).astype(np.float32)
print(f'  Q peak: {bc_tensor[:,:,1].max():.1f} m3/s  shape: {bc_tensor.shape}', flush=True)

# Step 5 — load warmstart pkl and patch only the BC fields
print('Patching warmstart pkl ...', flush=True)
train_pkl = os.path.join(OUT_ROOT, 'train', f'{DATASET_NAME}.pkl')
with open(train_pkl, 'rb') as f:
    data_out = pickle.load(f)[0]

data_out.node_BC        = torch.tensor(node_bc, dtype=torch.int32)
data_out.BC             = torch.FloatTensor(bc_tensor)
data_out.type_BC        = torch.tensor(2, dtype=torch.int32)
data_out.edge_BC_length = torch.ones(1, dtype=torch.float32)

print(f'  node_BC shape : {tuple(data_out.node_BC.shape)}', flush=True)
print(f'  BC shape      : {tuple(data_out.BC.shape)}', flush=True)
print(f'  type_BC       : {data_out.type_BC.item()} (discharge -> VX slot)', flush=True)
print(f'  Q peak        : {data_out.BC[:,:,1].max().item():.1f} m3/s', flush=True)

# Step 6 — save
print('Saving ...', flush=True)
for split in ('train', 'test'):
    out_dir  = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{DATASET_NAME}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump([copy.deepcopy(data_out)], f)
    print(f'  {out_path}', flush=True)

print('Done.', flush=True)
