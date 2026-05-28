"""Check BC structure in the warmstart pkl without needing CUDA."""
import sys, os
# Force CPU-only torch before any import
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    # Patch cuda before anything else loads it
    torch.cuda.is_available = lambda: False
except Exception as e:
    print(f'torch import failed: {e}')
    sys.exit(1)

import pickle

pkl_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'database', 'datasets', 'test',
    'ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart.pkl'
)

with open(pkl_path, 'rb') as f:
    d = pickle.load(f)[0]

n = d.node_BC.shape[0]
print(f'node_BC shape : {n}')
print(f'BC shape      : {tuple(d.BC.shape)}')
print(f'type_BC       : {d.type_BC.item()}')
if n >= 7:
    peak_inflow = float(d.BC[:7, :, 1].max())
    print(f'BC[:7] (inflow) peak WD: {peak_inflow:.3f} m  (expect ~2-4 m)')
if n > 7:
    peak_outlet = float(d.BC[7:, :, 1].max())
    print(f'BC[7:] (outlet) peak WD: {peak_outlet:.3f} m  (expect ~0 m)')
print(f'node_BC[:7]   : {d.node_BC[:7].tolist()}')
print(f'edge_BC_length: {d.edge_BC_length}')
