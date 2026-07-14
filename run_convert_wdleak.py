"""Convert: derive the 'WD-leak' dataset variant — the OLD forcing bug, deliberately restored.

Before the Q-forcing fix, the converter prescribed SFINCS ground-truth water depth
at the 7 source cells (type_BC=1) instead of the discharge hydrograph (type_BC=2).
This script recreates that dataset from the current warmstart pkl, for the
"before the fix" evidence run (plan point 3): train once on this dataset, then the
hydrograph-sensitivity test should give a flat line (the model never sees Q).

Usage (on hal8 or locally):  python run_convert_wdleak.py
(run_convert_* convention: converts/derives dataset pkls; it does NOT build a template —
it post-processes the existing warmstart pkl, so the 1.0x dataset must already exist.)
"""
import os
import pickle
import torch

NAME_IN  = 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart'
NAME_OUT = NAME_IN + '_wdleak'

for split in ['train', 'test']:
    src = os.path.join('database', 'datasets', split, NAME_IN + '.pkl')
    dst = os.path.join('database', 'datasets', split, NAME_OUT + '.pkl')

    print(f'[{split}] loading {src}')
    with open(src, 'rb') as f:
        dataset = pickle.load(f)

    for data in dataset:
        node_bc = data.node_BC.long()
        wd_at_bc = data.WD[node_bc, :]                       # GT water depth at the source cells
        BC = torch.zeros_like(data.BC)
        BC[:, :, 1] = wd_at_bc                               # dataset.py reads channel 1
        data.BC = BC
        data.type_BC = torch.tensor(1, dtype=torch.int32)    # 1 = water depth -> WD slot
        print(f'  node_BC={node_bc.tolist()}  WD@BC peak={wd_at_bc.max():.3f} m  type_BC=1')

    with open(dst, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'[{split}] saved {dst}')

print('\nDone. Train with config_best_sweep_wdleak_fixed_gt.yaml')
