#!/bin/bash
# Map hal8 jobs <-> W&B run IDs <-> configs <-> output checkpoints.
# Usage (on hal8, repo root):  bash list_runs.sh [N]
#   N = show only the last N jobs (default: all)
# Sources: logs/<job>_finetune.log (wandb offline dir => run id)
#          logs/<job>_finetune.out (Config: / Output: lines)

printf "%-8s %-10s %-9s %-45s %s\n" "JOB" "DATE" "WANDB_ID" "CONFIG" "OUTPUT"
printf "%s\n" "--------------------------------------------------------------------------------------------------"

for f in $(ls logs/*_finetune.log 2>/dev/null | sort -V); do
    j=$(basename "$f" _finetune.log)
    dir=$(grep -o "offline-run-[0-9_]*-[a-z0-9]*" "$f" | tail -1)
    id=${dir##*-}
    date=$(echo "$dir" | grep -o "[0-9]\{8\}" | head -1)
    out_file="logs/${j}_finetune.out"
    cfg=$(grep -m1 "^Config:" "$out_file" 2>/dev/null | awk '{print $2}')
    out=$(grep -m1 "^Output:" "$out_file" 2>/dev/null | awk '{print $2}')
    printf "%-8s %-10s %-9s %-45s %s\n" "$j" "${date:-?}" "${id:-?}" "${cfg:-?}" "${out:-?}"
done | { if [ -n "$1" ]; then tail -n "$1"; else cat; fi; }
