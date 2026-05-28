"""Check type_BC in training pkl files to understand what the model was trained with."""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle, glob

train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'datasets', 'train')
pkls = sorted(glob.glob(os.path.join(train_dir, '*.pkl')))

print(f'Found {len(pkls)} pkl files in train dir.\n')
for pkl_path in pkls[:8]:  # check first 8
    name = os.path.basename(pkl_path)
    try:
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)[0]
        type_bc = d.type_BC.item() if hasattr(d.type_BC, 'item') else int(d.type_BC)
        n_bc    = d.node_BC.shape[0] if hasattr(d, 'node_BC') else 'N/A'
        bc_shape = tuple(d.BC.shape) if hasattr(d, 'BC') else 'N/A'
        bc_max0  = float(d.BC[:, :, 0].max()) if hasattr(d, 'BC') else 'N/A'
        bc_max1  = float(d.BC[:, :, 1].max()) if hasattr(d, 'BC') else 'N/A'
        print(f'{name}')
        print(f'  type_BC={type_bc}  n_bc={n_bc}  BC_shape={bc_shape}')
        print(f'  BC[:,  :,0] max={bc_max0:.3f}   BC[:,:,1] max={bc_max1:.3f}')
    except Exception as e:
        print(f'{name}  ERROR: {e}')
    print()
