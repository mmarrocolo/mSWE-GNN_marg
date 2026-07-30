"""Wrapper: convert the per-location BC-augmentation simulations onto the 100 m template
and merge them, together with the existing uniform-scaling multisim events, into one
multi-sim training pkl.

Convention: build_* scripts create templates (build_template.py -> template_100m.pkl),
run_convert_* scripts project SFINCS outputs onto a template. Run build_template.py
first if the template is missing.

Background: the standard `multisim` training (run_convert_multisim.py) only ever scales all
7 BC source locations together, uniformly (0.5x/0.75x/1.25x/1.5x). The leave-one-out
sensitivity ladder in utils/visualize_bc_nonuniform_sensitivity.ipynb showed the resulting
model's response to a single location changing doesn't track that location's real volume
share (see the `bc-sensitivity-uniform-training-gap` memory note) -- these simulations
perturb one location (or a genuinely combined/desynchronized set) at a time, to teach the
model to distinguish each BC's individual effect rather than reacting only to aggregate
magnitude. Training keeps the existing q050-q150 uniform-scaling events too, so the model
doesn't lose what it already learned about whole-catchment magnitude scaling.

TRAIN_SIMS (16): one-at-a-time magnitude/timing perturbations, one all-locations-desynchronized
event, one catchment-emptying-from-baseline event, and one genuinely combined magnitude case
(Kirmutscheid x1.5 + Kreuzberg x0.5 simultaneously).

EXISTING_TRAIN_TAGS (4): the pre-existing q050/q075/q125/q150 uniform-scaling events from
run_convert_multisim.py, merged in alongside TRAIN_SIMS -- already converted, not reconverted
here.

TEST_SIMS (2): held out from training on purpose, to test *compositional* generalization --
each combines multiple simultaneous changes never presented together in TRAIN_SIMS:
  - desync2 (all-locations-desynchronized, every direction flipped vs. the training `alldesync`)
  - emptying_postpeak (draining from the flood-peak state instead of the pre-flood baseline)

Output:
  database/datasets/train|test/ahr_river_v03_marg_{tag}_additionalsrc_velocity_100m_warmstart.pkl
    for every TRAIN_SIMS/TEST_SIMS tag below (train+test written per scenario, per
    convert_sfincs_to_pkl_marg.py)
  database/datasets/train/ahr_river_v03_marg_bcaugment_additionalsrc_velocity_100m_warmstart_multisim.pkl
    (16 TRAIN_SIMS + 4 existing q050-q150 events merged into one list, for train_dataset_name)
  TEST_SIMS stay as separate individual test pkls -- evaluated one at a time in
  visualize_bc_nonuniform_sensitivity.ipynb, not merged.
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

TRAIN_SIMS = {
    # magnitude -- bidirectional (locations where the ladder showed the model is demonstrably broken)
    'kirmutscheid025x':      'ahr_river_v03_Marg_kirmutscheid025x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kirmutscheid2x':        'ahr_river_v03_Marg_kirmutscheid2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kreuzberg025x':         'ahr_river_v03_Marg_kreuzberg025x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kreuzberg2x':           'ahr_river_v03_Marg_kreuzberg2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'niederadenau025x':      'ahr_river_v03_Marg_niederadenau025x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'niederadenau2x':        'ahr_river_v03_Marg_niederadenau2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'new2_025x':             'ahr_river_v03_Marg_new2_025x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'new2_2x':               'ahr_river_v03_Marg_new2_2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    # magnitude -- single direction (unremarkable ladder response, lower priority)
    'denn2x':                'ahr_river_v03_Marg_denn2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'muesch2x':              'ahr_river_v03_Marg_muesch2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'new1_2x':               'ahr_river_v03_Marg_new1_2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    # timing -- Kirmutscheid solo, +-1.5x its own rise time (8.0h -> 12h)
    'kirmutscheid_shiftm12h': 'ahr_river_v03_Marg_kirmutscheid_shiftm12h_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'kirmutscheid_shiftp12h': 'ahr_river_v03_Marg_kirmutscheid_shiftp12h_additionalsrc_velocity_100m_cutpolygon_warmstart',
    # all-locations-desynchronized (timing) and catchment-emptying-from-baseline
    'alldesync':             'ahr_river_v03_Marg_alldesync_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'alldrain':              'ahr_river_v03_Marg_alldrain_additionalsrc_velocity_100m_cutpolygon_warmstart',
    # genuinely combined magnitude perturbation (kept in training, not held out)
    'kirmutscheid15x_kreuzberg05x': 'ahr_river_v03_Marg_kirmutscheid15x_kreuzberg05x_additionalsrc_velocity_100m_cutpolygon_warmstart',
}

TEST_SIMS = {
    # held out on purpose -- compositional generalization checks, see module docstring
    'desync2':           'ahr_river_v03_Marg_desync2_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'emptying_postpeak': 'ahr_river_v03_Marg_emptying_postpeak_additionalsrc_velocity_100m_cutpolygon_warmstart',
}

# Existing uniform-scaling events from run_convert_multisim.py, already converted under a
# different naming pattern (no per-location tag) -- merged in alongside TRAIN_SIMS below,
# not reconverted here.
EXISTING_TRAIN_TAGS = ['q050', 'q075', 'q125', 'q150']
EXISTING_PER_SIM_NAME = 'ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_{tag}'

PER_SIM_NAME = 'ahr_river_v03_marg_{tag}_additionalsrc_velocity_100m_warmstart'
MERGED_NAME  = 'ahr_river_v03_marg_bcaugment_additionalsrc_velocity_100m_warmstart_multisim'

assert os.path.exists(TEMPLATE_PKL), \
    f'Template not found: {TEMPLATE_PKL} -- run build_template.py first'

spec = importlib.util.spec_from_file_location(
    'convert_sfincs_to_pkl_marg',
    os.path.join(PROJECT_ROOT, 'database', 'convert_sfincs_to_pkl_marg.py'),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def convert_one(tag, folder):
    sim_dir = os.path.join(SIM_ROOT, folder)
    for fname in ['sfincs_map.nc', 'sfincs.src', 'sfincs.dis']:
        assert os.path.exists(os.path.join(sim_dir, fname)), f'{tag}: missing {fname} in {sim_dir}'

    out_train = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                             PER_SIM_NAME.format(tag=tag) + '.pkl')
    if os.path.exists(out_train):
        print(f'skipping {tag} (already converted): {out_train}')
        return

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


# --- 1. convert every training + held-out test simulation onto the template ---
for tag, folder in {**TRAIN_SIMS, **TEST_SIMS}.items():
    convert_one(tag, folder)

# --- 2. merge TRAIN_SIMS + the existing q050-q150 events into one multi-sim training pkl ---
merged = []
for tag in TRAIN_SIMS:
    pkl_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                            PER_SIM_NAME.format(tag=tag) + '.pkl')
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    for data in data_list:
        print(f'{tag}: node_BC = {data.node_BC.tolist()}   WD peak = {float(data.WD.max()):.3f} m   '
              f'n_steps = {data.WD.shape[1]}')
    merged += data_list

for tag in EXISTING_TRAIN_TAGS:
    pkl_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                            EXISTING_PER_SIM_NAME.format(tag=tag) + '.pkl')
    assert os.path.exists(pkl_path), \
        f'{tag}: expected pre-existing pkl not found at {pkl_path} -- run run_convert_multisim.py first'
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    for data in data_list:
        print(f'{tag} (existing): node_BC = {data.node_BC.tolist()}   WD peak = {float(data.WD.max()):.3f} m   '
              f'n_steps = {data.WD.shape[1]}')
    merged += data_list

out_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', MERGED_NAME + '.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(merged, f)
print(f'\nSaved {len(merged)} training events '
      f'({len(TRAIN_SIMS)} BC-augmentation + {len(EXISTING_TRAIN_TAGS)} existing uniform-scaling) -> {out_path}')
print('TEST_SIMS were converted but NOT merged -- evaluate them individually, e.g. via '
      'evaluate_new_scenario() in visualize_bc_nonuniform_sensitivity.ipynb.')
print('Note: alldrain runs far longer (~450 steps) than the others (~120) -- to_temporal_dataset '
      'will generate proportionally more training samples from it; worth checking this does not '
      'over-represent draining dynamics in the resulting training mix.')
