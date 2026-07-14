"""Wrapper: convert the 4 scaled-hydrograph warmstart simulations (Linea 2) onto the
100 m template and merge them into one multi-sim training pkl.

Convention: build_* scripts create templates (build_template.py -> template_100m.pkl),
run_convert_* scripts project SFINCS outputs onto a template. Run build_template.py
first if the template is missing.

Output:
  database/datasets/train|test/ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_{q050,q075,q125,q150}.pkl
  database/datasets/train/ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_multisim.pkl  (4 events)

The 1.0x event stays out (it is the existing ..._warmstart dataset, used as
validation/test by config_best_sweep_multisim.yaml with validate_on_test: True).
"""
import importlib.util
import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEMPLATE_PKL = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', 'template_100m.pkl')
SIM_ROOT     = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations')

SIMS = {
    'q050': 'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart_q050',
    'q075': 'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart_q075',
    'q125': 'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart_q125',
    'q150': 'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart_q150',
}
PER_SIM_NAME = 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_{tag}'
MERGED_NAME  = 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_multisim'
REF_1X_PKL   = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                            'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart.pkl')

assert os.path.exists(TEMPLATE_PKL), \
    f'Template not found: {TEMPLATE_PKL} — run build_template.py first'

spec = importlib.util.spec_from_file_location(
    'convert_sfincs_to_pkl_marg',
    os.path.join(PROJECT_ROOT, 'database', 'convert_sfincs_to_pkl_marg.py'),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# --- 1. convert each scaled simulation onto the template ---
for tag, folder in SIMS.items():
    sim_dir = os.path.join(SIM_ROOT, folder)
    for fname in ['sfincs_map.nc', 'sfincs.src', 'sfincs.dis']:
        assert os.path.exists(os.path.join(sim_dir, fname)), f'{tag}: missing {fname} in {sim_dir}'

    out_train = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                             PER_SIM_NAME.format(tag=tag) + '.pkl')
    if os.path.exists(out_train):
        print(f'skipping {tag} (already converted): {out_train}')
        continue

    print(f'\n=== converting {tag} ({folder}) ===')
    sys.argv = [
        'convert_sfincs_to_pkl_marg.py',
        '--sfincs-map',   os.path.join(sim_dir, 'sfincs_map.nc'),
        '--template-pkl', TEMPLATE_PKL,
        '--dataset-name', PER_SIM_NAME.format(tag=tag),
        '--out-root',     os.path.join(PROJECT_ROOT, 'database', 'datasets'),
        '--vx-var', 'u',
        '--vy-var', 'v',
        '--src-file', os.path.join(sim_dir, 'sfincs.src'),
        '--dis-file', os.path.join(sim_dir, 'sfincs.dis'),
    ]
    mod.main()

# --- 2. merge the 4 events into one training pkl ---
with open(REF_1X_PKL, 'rb') as f:
    ref_1x = pickle.load(f)
q_ref = float(ref_1x[0].BC[:, :, 1].max())
node_bc_ref = ref_1x[0].node_BC.tolist()
print(f'\n1.0x reference: Q peak = {q_ref:.1f} m3/s   node_BC = {node_bc_ref}')

merged = []
for tag in SIMS:
    pkl_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                            PER_SIM_NAME.format(tag=tag) + '.pkl')
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    for data in data_list:
        q_peak = float(data.BC[:, :, 1].max())
        same_bc = data.node_BC.tolist() == node_bc_ref
        print(f'{tag}: Q peak = {q_peak:8.1f} m3/s (ratio {q_peak/q_ref:.3f})   '
              f'WD peak = {float(data.WD.max()):.3f} m   node_BC identical: {same_bc}')
        assert same_bc, f'{tag}: node_BC differs from the 1.0x event!'
    merged += data_list

out_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', MERGED_NAME + '.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(merged, f)
print(f'\nSaved {len(merged)} events -> {out_path}')
print('Sanity: Q-peak ratios above must read ~0.5 / 0.75 / 1.25 / 1.5.')
print('Train with config_best_sweep_multisim.yaml (validation/test = held-out 1.0x event).')
