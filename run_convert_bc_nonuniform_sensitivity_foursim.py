"""Wrapper: convert the 7 real SFINCS ground-truth runs for the non-uniform BC sensitivity test
(utils/visualize_bc_nonuniform_sensitivity_foursim.ipynb) onto the 100 m template.

Background: that notebook's `scenarios` block (cell 9) only ever fakes 6 of these 7 scenarios by
scaling the single real baseline event's BC tensor and rolling the model out on it -- ground truth
(`gt_s0`) stays the real baseline event for every row, so it can compare the model's *own* response
across scenarios but can't check absolute accuracy or flood-peak timing per scenario. To check for a
lagged/wrong flood peak (i.e. a bad water-residence-time behavior in the catchment) each scenario
needs its own dedicated SFINCS ground truth to compare the GNN rollout against -- that's what these
7 runs are for.

Convention: build_* scripts create templates (build_template.py -> template_100m.pkl),
run_convert_* scripts project SFINCS outputs onto a template. Run build_template.py
first if the template is missing.

Output (test-only GT, like TEST_SIMS in run_convert_bc_augmentation.py -- not merged into a
training multisim pkl, evaluated one at a time via evaluate_new_scenario() in the notebook):
  database/datasets/train|test/ahr_river_v03_marg_{tag}_additionalsrc_velocity_100m_warmstart.pkl
    for each tag below (train+test written per scenario, per convert_sfincs_to_pkl_marg.py's
    inference-only save path)
"""
import importlib.util
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEMPLATE_PKL = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', 'template_100m.pkl')
SIM_ROOT     = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations')

# Tags match the notebook's `scenarios` dict keys (cell 9) exactly, so the resulting pkl names
# can be passed straight to evaluate_new_scenario().
GT_SIMS = {
    'baseline_1x':             'ahr_river_v03_Marg_baseline_1x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'tributaries_0x':          'ahr_river_v03_Marg_tributaries_0x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'tributaries_2x':          'ahr_river_v03_Marg_tributaries_2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'mainstem_0x':             'ahr_river_v03_Marg_mainstem_0x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'mainstem_2x':             'ahr_river_v03_Marg_mainstem_2x_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'only_mainstem':           'ahr_river_v03_Marg_only_mainstem_additionalsrc_velocity_100m_cutpolygon_warmstart',
    'only_smallest_tributary': 'ahr_river_v03_Marg_only_smallest_tributary_additionalsrc_velocity_100m_cutpolygon_warmstart',
}

PER_SIM_NAME = 'ahr_river_v03_marg_{tag}_additionalsrc_velocity_100m_warmstart'

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


for tag, folder in GT_SIMS.items():
    convert_one(tag, folder)

print('\nDone. Evaluate each with evaluate_new_scenario(dataset_name) in '
      'utils/visualize_bc_nonuniform_sensitivity_foursim.ipynb, e.g.:')
for tag in GT_SIMS:
    print(f"  {PER_SIM_NAME.format(tag=tag)}")
