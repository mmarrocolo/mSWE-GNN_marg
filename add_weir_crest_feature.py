"""Patch edge_weir_crest onto the 16 already-converted structures pkls, in place.

Adds a new edge feature only - does NOT touch WD/VX/VY/DEM/mesh, so this does not
need to re-run the SFINCS interpolation (interpolate_time_series/scipy.griddata,
the crash-prone step on this machine). Safe to run locally.

For each of the 16 TRAIN_SIMS/TEST_SIMS tags, reads that scenario's own
sfincs.weir (crest elevation per weir line) and computes, per dual edge,
max(0, crest_z - local terrain), 0 everywhere there is no structure - see
compute_edge_weir_crest_offset in database/convert_sfincs_to_pkl_marg.py.
Patches both the train/ and test/ copies of each per-scenario pkl (main() in
convert_sfincs_to_pkl_marg.py writes identical copies to both).

Run this, then re-run Step 3 of create_dataset_structures.ipynb (or the merge
block at the bottom of run_convert_structures.py) to rebuild the merged
32-event training pkl from the patched TRAIN_SIMS pkls + the untouched 20
existing BC-augmentation events (which have no edge_weir_crest attribute at
all - get_edge_features() falls back to all-zero for those, see dataset.py).
"""
import os
import pickle
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATABASE_DIR = os.path.join(PROJECT_ROOT, 'database')
if DATABASE_DIR not in sys.path:
    sys.path.insert(0, DATABASE_DIR)

from convert_sfincs_to_pkl_marg import parse_weir_file, compute_edge_weir_crest_offset

SIM_ROOT = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations')
SIM_FOLDER = 'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart_infra_{tag}'
PER_SIM_NAME = 'ahr_river_v03_marg_structures_{tag}_additionalsrc_velocity_100m_warmstart'
DATASETS_ROOT = os.path.join(PROJECT_ROOT, 'database', 'datasets')

ALL_TAGS = [
    'bothbanks_dz2m', 'bothbanks_dz5m', 'bothbanks_dz10m',
    'northonly_dz2m', 'northonly_dz5m',
    'southonly_dz2m', 'southonly_dz5m',
    'upstream_dz2m', 'upstream_dz5m',
    'downstream_dz2m', 'downstream_dz5m',
    'gap_dz5m',
    'thin_dam_dz15m', 'thin_dam_dz20m',
    'thin_dam2_dz15m', 'thin_dam2_dz20m',
]

for tag in ALL_TAGS:
    folder = SIM_FOLDER.format(tag=tag)
    weir_path = os.path.join(SIM_ROOT, folder, 'sfincs.weir')
    final_name = PER_SIM_NAME.format(tag=tag)

    if not os.path.exists(weir_path):
        print(f'{tag}: no sfincs.weir found at {weir_path} - skipping')
        continue

    weirs = parse_weir_file(weir_path)
    print(f'\n{tag}: {len(weirs)} weir line(s) from {weir_path}')

    for split in ('train', 'test'):
        pkl_path = os.path.join(DATASETS_ROOT, split, final_name + '.pkl')
        if not os.path.exists(pkl_path):
            print(f'  [skip] {pkl_path} not found')
            continue

        with open(pkl_path, 'rb') as f:
            data_list = pickle.load(f)
        data = data_list[0]

        offset = compute_edge_weir_crest_offset(data, weirs)
        data.edge_weir_crest = torch.FloatTensor(offset)
        n_tagged = int((data.edge_weir_crest > 0).sum())
        print(f'  {split}: tagged {n_tagged} dual edges '
              f'(max offset {float(data.edge_weir_crest.max()):.2f} m) -> {pkl_path}')

        with open(pkl_path, 'wb') as f:
            pickle.dump(data_list, f)

print('\nDone. Now re-run Step 3 (merge) to rebuild the 32-event training pkl.')
