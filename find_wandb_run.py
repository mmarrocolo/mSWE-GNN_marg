"""Find the wandb run that came from a given hal8 SLURM job.

Usage (paste on hal8, from the repo root):
    python find_wandb_run.py --job_id "$SLURM_JOB_ID"
    python find_wandb_run.py --job_id 243256

Works two ways, tried in order:
  1. Local offline-run folders under wandb/ (wandb/offline-run-*/files/config.yaml).
     No network needed - works even before you've run `wandb sync`.
  2. The wandb API, searching the mswe-gnn-ahr-finetune project for a run whose
     config.slurm_job_id matches - only useful after `wandb sync`, and only from a
     node with internet access (e.g. the v-slurmsub001 submission node, not a
     compute node).

Only finds runs launched by finetune_ahr.py AFTER the slurm_job_id field was added
(see finetune_ahr.py's wandb.init call) - older synced runs predate this and won't
match either way.
"""
import argparse
import glob
import os

import yaml

PROJECT = "mswe-gnn-ahr-finetune"


def _unwrap(v):
    """wandb's saved config.yaml wraps each value as {'value': ..., 'desc': ...}."""
    return v.get('value', v) if isinstance(v, dict) else v


def search_local(job_id):
    matches = []
    for cfg_path in glob.glob(os.path.join('wandb', 'offline-run-*', 'files', 'config.yaml')):
        try:
            with open(cfg_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            continue
        if str(_unwrap(cfg.get('slurm_job_id', {}))) == str(job_id):
            run_dir = os.path.dirname(os.path.dirname(cfg_path))
            run_id = run_dir.rsplit('-', 1)[-1]
            matches.append({
                'run_dir': run_dir,
                'run_id': run_id,
                'run_name': _unwrap(cfg.get('_wandb', {})),  # not always present; best effort
                'config_file': _unwrap(cfg.get('config_file', {})),
            })
    return matches


def search_api(job_id):
    try:
        import wandb
        api = wandb.Api()
        entity = api.default_entity
        if entity is None:
            return []
        runs = api.runs(f"{entity}/{PROJECT}", filters={"config.slurm_job_id": str(job_id)})
        return [{'name': r.name, 'id': r.id, 'url': r.url} for r in runs]
    except Exception as e:
        print(f"  (wandb API search skipped: {e})")
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job_id', required=True, help='SLURM job ID, e.g. "$SLURM_JOB_ID" or 243256')
    args = p.parse_args()

    print(f"Looking for the wandb run from SLURM job {args.job_id}...")

    local = search_local(args.job_id)
    if local:
        print(f"\nFound {len(local)} local offline-run folder(s):")
        for m in local:
            print(f"  run_id={m['run_id']}  config_file={m['config_file']}  dir={m['run_dir']}")
    else:
        print("\nNo match in local wandb/offline-run-* folders.")

    api_matches = search_api(args.job_id)
    if api_matches:
        print(f"\nFound {len(api_matches)} run(s) on wandb.ai:")
        for m in api_matches:
            print(f"  {m['name']}  ({m['id']})  {m['url']}")
    else:
        print("No match via the wandb API (either not synced yet, no network here, "
              "or this run predates the slurm_job_id field).")


if __name__ == '__main__':
    main()
