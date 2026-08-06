"""Wrapper: convert the levee/dam structure simulations onto the 100 m structures template
and merge the training ones, together with the existing BC-augmentation multisim events,
into one multi-sim training pkl.

Convention: build_* scripts create templates (build_template_structures.py ->
template_100m_structures.pkl), run_convert_* scripts project SFINCS outputs onto a
template. Run build_template_structures.py first if the template is missing.

TRAIN_SIMS (12): levee sweep across 5 sites (bothbanks, northonly, southonly, upstream,
downstream) x 2 crest heights (dz2m, dz5m), plus one thin-dam crossing at 2 crest
heights (dz15m, dz20m).

TEST_SIMS (4), held out on purpose, three distinct generalization axes:
  - bothbanks_dz10m: unseen magnitude (2x beyond the highest trained levee crest)
  - gap_dz5m: unseen structural footprint (a breach, same site family as the trained levees)
  - thin_dam2_dz15m / thin_dam2_dz20m: unseen location for the thin-dam mechanism
    (~3-4 km from the trained thin_dam crossing), same crest heights as TRAIN_SIMS

EXISTING_TRAIN_PKL: the already-converted BC-augmentation multisim training pkl (20 events,
built onto template_100m.pkl by run_convert_bc_augmentation.py), merged in alongside
TRAIN_SIMS so the model keeps training on BC-magnitude/timing diversity as well as
structure scenarios. Different mesh template than TRAIN_SIMS, but each Data object
carries its own edge_index/mesh, so mixing them in one training list is fine.

Output:
  database/datasets/train|test/ahr_river_v03_marg_structures_{tag}_additionalsrc_velocity_100m_warmstart.pkl
    for every TRAIN_SIMS/TEST_SIMS tag below
  database/datasets/train/ahr_river_v03_marg_structures_bcaugment_additionalsrc_velocity_100m_warmstart_multisim.pkl
    (12 TRAIN_SIMS + 20 existing BC-augmentation events merged into one list, for train_dataset_name)
  TEST_SIMS stay as separate individual test pkls -- evaluate them one at a time.
"""
import os
import pickle
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# The partial/merged pkls carry Mesh objects pickled with module path bare
# 'graph_creation' (see build_template_structures.py comments). This script does its
# own pickle.load calls (merge step below), so needs database/ on sys.path too - the
# convert_sfincs_to_pkl_marg.py subprocess calls get this for free since Python adds
# a script's own directory to sys.path automatically when run directly.
DATABASE_DIR = os.path.join(PROJECT_ROOT, 'database')
if DATABASE_DIR not in sys.path:
    sys.path.insert(0, DATABASE_DIR)

TEMPLATE_PKL = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', 'template_100m_structures.pkl')
SIM_ROOT     = os.path.join(PROJECT_ROOT, 'database', 'raw_datasets_ahr', 'Simulations')

SIM_FOLDER = 'ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart_infra_{tag}'

TRAIN_SIMS = {
    tag: SIM_FOLDER.format(tag=tag) for tag in [
        'bothbanks_dz2m', 'bothbanks_dz5m',
        'northonly_dz2m', 'northonly_dz5m',
        'southonly_dz2m', 'southonly_dz5m',
        'upstream_dz2m', 'upstream_dz5m',
        'downstream_dz2m', 'downstream_dz5m',
        'thin_dam_dz15m', 'thin_dam_dz20m',
    ]
}

TEST_SIMS = {
    tag: SIM_FOLDER.format(tag=tag) for tag in [
        'bothbanks_dz10m',    # unseen magnitude
        'gap_dz5m',            # unseen structural footprint
        'thin_dam2_dz15m',     # unseen location, same mechanism
        'thin_dam2_dz20m',
    ]
}

# Already-converted BC-augmentation multisim training pkl (20 events, on template_100m.pkl)
EXISTING_TRAIN_PKL = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train',
                                   'ahr_river_v03_marg_bcaugment_additionalsrc_velocity_100m_warmstart_multisim.pkl')

PER_SIM_NAME = 'ahr_river_v03_marg_structures_{tag}_additionalsrc_velocity_100m_warmstart'
MERGED_NAME  = 'ahr_river_v03_marg_structures_bcaugment_additionalsrc_velocity_100m_warmstart_multisim'

assert os.path.exists(TEMPLATE_PKL), \
    f'Template not found: {TEMPLATE_PKL} -- run build_template_structures.py first'
assert os.path.exists(EXISTING_TRAIN_PKL), \
    f'Existing BC-augmentation multisim pkl not found: {EXISTING_TRAIN_PKL} -- run run_convert_bc_augmentation.py first'

CONVERT_SCRIPT = os.path.join(PROJECT_ROOT, 'database', 'convert_sfincs_to_pkl_marg.py')


def _run_convert(dataset_name, sim_dir, only_var=None, max_attempts=3):
    """Runs convert_sfincs_to_pkl_marg.py as its own subprocess, with retries."""
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


def convert_one(tag, folder):
    sim_dir = os.path.join(SIM_ROOT, folder)
    for fname in ['sfincs_map.nc', 'sfincs.src', 'sfincs.dis']:
        assert os.path.exists(os.path.join(sim_dir, fname)), f'{tag}: missing {fname} in {sim_dir}'

    final_name = PER_SIM_NAME.format(tag=tag)
    out_train = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', final_name + '.pkl')
    if os.path.exists(out_train):
        print(f'skipping {tag} (already converted): {out_train}')
        return

    print(f'\n=== converting {tag} ({folder}) ===')
    # Running convert_sfincs_to_pkl_marg.py's full WD+VX+VY pipeline in one process
    # (~360 griddata/Qhull calls) crashes unpredictably on this machine. A lone run
    # of just one variable (~120 calls) is reliable, so compute WD/VX/VY as three
    # separate subprocess invocations (via --only-var), each spawned fresh from this
    # script - which never imports torch, unlike convert_sfincs_to_pkl_marg.py itself,
    # so the spawn itself is also the reliable case (see project_local_gpu_no_cuda
    # memory for both crash classes). Then merge the three partial results into one
    # final Data object.
    partial_names = {var: f'{final_name}_partial_{var}' for var in ['WD', 'VX', 'VY']}
    for var, name in partial_names.items():
        _run_convert(name, sim_dir, only_var=var)

    import pickle as _pickle
    partial_train = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train')
    partial_test = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'test')

    with open(os.path.join(partial_train, partial_names['WD'] + '.pkl'), 'rb') as f:
        data = _pickle.load(f)[0]
    with open(os.path.join(partial_train, partial_names['VX'] + '.pkl'), 'rb') as f:
        data.VX = _pickle.load(f)[0].VX
    with open(os.path.join(partial_train, partial_names['VY'] + '.pkl'), 'rb') as f:
        data.VY = _pickle.load(f)[0].VY

    out_test = os.path.join(partial_test, final_name + '.pkl')
    with open(out_train, 'wb') as f:
        _pickle.dump([data], f)
    with open(out_test, 'wb') as f:
        _pickle.dump([data], f)
    print(f'  {tag}: merged WD/VX/VY -> {out_train}')

    for name in partial_names.values():
        for split_dir in (partial_train, partial_test):
            p = os.path.join(split_dir, name + '.pkl')
            if os.path.exists(p):
                os.remove(p)


# --- 1. convert every training + held-out test simulation onto the structures template ---
for tag, folder in {**TRAIN_SIMS, **TEST_SIMS}.items():
    convert_one(tag, folder)

# --- 2. merge TRAIN_SIMS + the existing bc-augmentation multisim events into one training pkl ---
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

with open(EXISTING_TRAIN_PKL, 'rb') as f:
    existing = pickle.load(f)
print(f'\nMerging in {len(existing)} existing BC-augmentation events from {EXISTING_TRAIN_PKL}')
merged += existing

out_path = os.path.join(PROJECT_ROOT, 'database', 'datasets', 'train', MERGED_NAME + '.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(merged, f)
print(f'\nSaved {len(merged)} training events '
      f'({len(TRAIN_SIMS)} structure scenarios + {len(existing)} existing BC-augmentation events) -> {out_path}')
print('TEST_SIMS were converted but NOT merged -- evaluate them individually for generalization checks.')
