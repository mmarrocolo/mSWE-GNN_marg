"""Reconvert the 20 BC-augmentation simulations onto template_100m.pkl, in place under
their original filenames.

Why: these 20 per-sim pkls (16 TRAIN_SIMS + 4 existing q050-q150 events, see
run_convert_bc_augmentation.py) were converted before template_100m.pkl carried
node_BC_outflow / bidirectional absorbing-outflow ghost edges. They were deliberately
left stale at the time (re-running 18+ simulations was expensive) - see the
template-bc-bugs-fixed-aug2026 memory note. template_100m.pkl itself has since been
verified to already have node_BC_outflow (87 cells, bidirectional edges, correct
database.graph_creation class identity) - same topology (30,079 nodes), so this is a
straight reconversion onto the SAME template, not a template change.

Motivation: mixing these (no node_BC_outflow) with the 12 new structure-scenario events
(which do have it) in one training set means ~2/3 of the merged training data never
gets the absorbing-outflow clamp applied at all - a real risk of the model learning to
key off which "family" a sample belongs to rather than a genuine, transferable drainage
rule. Reconverting these 20 the same way removes that split.

Output: overwrites, in place, the same 20 filenames run_convert_bc_augmentation.py
produces, plus the same merged
ahr_river_v03_marg_bcaugment_additionalsrc_velocity_100m_warmstart_multisim.pkl name -
so nothing downstream needs to change, it just becomes correct. Re-run
run_convert_structures.py afterward to rebuild the final 32-event structures+bcaugment
merge from this corrected input (it will skip the 12 already-converted structure sims
and just redo the merge step).
"""
import os
import pickle
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATABASE_DIR = os.path.join(PROJECT_ROOT, 'database')
if DATABASE_DIR not in sys.path:
    sys.path.insert(0, DATABASE_DIR)

TEMPLATE_PKL = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', 'template_100m.pkl')
SIM_ROOT     = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations')

TRAIN_SIMS = {
    # 0x (no-flow) tributary runs replace the earlier 025x ones for these 4 tributaries
    # (2026-08-06) - new simulations added to the Simulations folder.
    'kirmutscheid0x':        'ahr_river_v03_Marg_kirmutscheid0x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kirmutscheid2x':        'ahr_river_v03_Marg_kirmutscheid2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kreuzberg0x':           'ahr_river_v03_Marg_kreuzberg0x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kreuzberg2x':           'ahr_river_v03_Marg_kreuzberg2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'niederadenau0x':        'ahr_river_v03_Marg_niederadenau0x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'niederadenau2x':        'ahr_river_v03_Marg_niederadenau2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'new2_0x':               'ahr_river_v03_Marg_new2_0x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'new2_2x':               'ahr_river_v03_Marg_new2_2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'denn2x':                'ahr_river_v03_Marg_denn2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'muesch2x':              'ahr_river_v03_Marg_muesch2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'new1_2x':               'ahr_river_v03_Marg_new1_2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kirmutscheid_shiftm12h': 'ahr_river_v03_Marg_kirmutscheid_shiftm12h_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kirmutscheid_shiftp12h': 'ahr_river_v03_Marg_kirmutscheid_shiftp12h_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'alldesync':             'ahr_river_v03_Marg_alldesync_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'alldrain':              'ahr_river_v03_Marg_alldrain_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kirmutscheid15x_kreuzberg05x': 'ahr_river_v03_Marg_kirmutscheid15x_kreuzberg05x_additionalsrc_velocity_100m_cutpolygon_warmstart',
}
PER_SIM_NAME = 'ahr_river_v03_marg_{tag}_additionalsrc_velocity_100m_warmstart'

EXISTING_TRAIN_TAGS = ['q050', 'q075', 'q125', 'q150']
EXISTING_SIM_FOLDER = 'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart_{tag}'
EXISTING_PER_SIM_NAME = 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_{tag}'

MERGED_NAME = 'ahr_river_v03_marg_bcaugment_additionalsrc_velocity_100m_warmstart_multisim'

assert os.path.exists(TEMPLATE_PKL), f'Template not found: {TEMPLATE_PKL}'

CONVERT_SCRIPT = os.path.join(PROJECT_ROOT, 'database', 'convert_sfincs_to_pkl_marg.py')


def _run_convert(dataset_name, sim_dir, only_var=None, max_attempts=15):
    # alldrain runs ~450 timesteps (vs ~121 for the others) - ~4x more sequential
    # griddata/Qhull calls per variable, so it hits the native-crash flakiness far more
    # often per attempt. Higher retry budget here to compensate.
    args = [
        sys.executable, CONVERT_SCRIPT,
        '--sfincs-map',   os.path.join(sim_dir, 'sfincs_map.nc'),
        '--template-pkl', TEMPLATE_PKL,
        '--dataset-name', dataset_name,
        '--out-root',     os.path.join(PROJECT_ROOT, 'database', 'datasets'),
        '--vx-var', 'u',
        '--vy-var', 'v',
        '--src-file', os.path.join(sim_dir, 'sfincs.src'),
        '--dis-file', os.path.join(sim_dir, 'sfincs.dis'),
    ]
    if only_var is not None:
        args += ['--only-var', only_var]
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(args, env=env)
        if result.returncode == 0:
            return
        print(f'  {dataset_name}: conversion crashed (attempt {attempt}/{max_attempts}, exit {result.returncode}); retrying...')
    raise RuntimeError(f'{dataset_name}: conversion failed after {max_attempts} attempts')


def reconvert_one(final_name, folder):
    sim_dir = os.path.join(SIM_ROOT, folder)
    for fname in ['sfincs_map.nc', 'sfincs.src', 'sfincs.dis']:
        assert os.path.exists(os.path.join(sim_dir, fname)), f'{final_name}: missing {fname} in {sim_dir}'

    out_train = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', final_name + '.pkl')
    if os.path.exists(out_train):
        print(f'skipping {final_name} (already reconverted): {out_train}')
        return

    print(f'\n=== reconverting {final_name} ({folder}) ===')
    partial_names = {var: f'{final_name}_partial_{var}' for var in ['WD', 'VX', 'VY']}
    for var, name in partial_names.items():
        _run_convert(name, sim_dir, only_var=var)

    partial_train = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train')
    partial_test = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'test')

    with open(os.path.join(partial_train, partial_names['WD'] + '.pkl'), 'rb') as f:
        data = pickle.load(f)[0]
    with open(os.path.join(partial_train, partial_names['VX'] + '.pkl'), 'rb') as f:
        data.VX = pickle.load(f)[0].VX
    with open(os.path.join(partial_train, partial_names['VY'] + '.pkl'), 'rb') as f:
        data.VY = pickle.load(f)[0].VY

    out_test = os.path.join(partial_test, final_name + '.pkl')
    with open(out_train, 'wb') as f:
        pickle.dump([data], f)
    with open(out_test, 'wb') as f:
        pickle.dump([data], f)
    print(f'  {final_name}: merged WD/VX/VY -> {out_train}   node_BC_outflow={hasattr(data, "node_BC_outflow")}')

    for name in partial_names.values():
        for split_dir in (partial_train, partial_test):
            p = os.path.join(split_dir, name + '.pkl')
            if os.path.exists(p):
                os.remove(p)


# --- 1. reconvert all 16 TRAIN_SIMS ---
for tag, folder in TRAIN_SIMS.items():
    reconvert_one(PER_SIM_NAME.format(tag=tag), folder)

# --- 2. reconvert all 4 existing q050-q150 events ---
for tag in EXISTING_TRAIN_TAGS:
    reconvert_one(EXISTING_PER_SIM_NAME.format(tag=tag), EXISTING_SIM_FOLDER.format(tag=tag))

# --- 3. re-merge, overwriting the same merged filename in place ---
merged = []
for tag in TRAIN_SIMS:
    pkl_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                            PER_SIM_NAME.format(tag=tag) + '.pkl')
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    for data in data_list:
        print(f'{tag}: node_BC = {data.node_BC.tolist()}   node_BC_outflow={hasattr(data, "node_BC_outflow")}   '
              f'WD peak = {float(data.WD.max()):.3f} m   n_steps = {data.WD.shape[1]}')
    merged += data_list

for tag in EXISTING_TRAIN_TAGS:
    pkl_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                            EXISTING_PER_SIM_NAME.format(tag=tag) + '.pkl')
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    for data in data_list:
        print(f'{tag} (existing): node_BC = {data.node_BC.tolist()}   node_BC_outflow={hasattr(data, "node_BC_outflow")}   '
              f'WD peak = {float(data.WD.max()):.3f} m   n_steps = {data.WD.shape[1]}')
    merged += data_list

out_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', MERGED_NAME + '.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(merged, f)
print(f'\nSaved {len(merged)} training events (all with node_BC_outflow now) -> {out_path}')
print('Next: re-run run_convert_structures.py to rebuild the 32-event structures+bcaugment merge from this.')
