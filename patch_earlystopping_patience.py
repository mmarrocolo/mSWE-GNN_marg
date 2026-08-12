"""
Patch a Lightning checkpoint's saved EarlyStopping callback state so that a
`--resume` continuation actually gets more patience, instead of immediately
re-triggering.

Why this is needed: `trainer.fit(ckpt_path=...)` restores the EarlyStopping
callback's *entire* saved state on resume, including `patience` itself (not
just `wait_count`) - see lightning/pytorch/callbacks/early_stopping.py,
state_dict()/load_state_dict(). Bumping `patience` in the yaml config has no
effect on a resumed run: Lightning loads the checkpoint's old patience value
right back over it, and wait_count is restored already at (or past) the old
threshold, so training stops again within the first post-resume validation
check. This script edits the checkpoint's saved EarlyStopping state directly
so the resume is genuine.

Usage (run on hal8, from the repo root, before resuming):
    python patch_earlystopping_patience.py <path/to/last.ckpt> --patience 100

Patches in place by default (writes a .bak backup first). Use --out to write
a separate file instead of touching the original.
"""
import argparse
import shutil

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_path")
    p.add_argument("--patience", type=int, required=True)
    p.add_argument("--monitor", default="val_loss")
    p.add_argument("--mode", default="min")
    p.add_argument("--out", default=None, help="Write to a new path instead of patching in place")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    callbacks = ckpt.get("callbacks", {})

    state_key = f"EarlyStopping{dict(monitor=args.monitor, mode=args.mode)!r}"
    if state_key not in callbacks:
        raise SystemExit(
            f"EarlyStopping state key not found: {state_key!r}\n"
            f"Available callback keys: {list(callbacks.keys())}"
        )

    es_state = callbacks[state_key]
    old_wait, old_patience = es_state["wait_count"], es_state["patience"]
    es_state["wait_count"] = 0
    es_state["patience"] = args.patience
    es_state["stopped_epoch"] = 0
    es_state["stopping_reason"] = 0  # EarlyStoppingReason.NOT_STOPPED

    out_path = args.out or args.ckpt_path
    if args.out is None:
        shutil.copy(args.ckpt_path, args.ckpt_path + ".bak")
        print(f"Backed up original to {args.ckpt_path}.bak")

    torch.save(ckpt, out_path)
    print(f"wait_count {old_wait} -> 0, patience {old_patience} -> {args.patience}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
