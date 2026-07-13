#!/bin/bash
# MASTER baseline — run CSI300 (CN) and SP500 (US).
#
# Usage (from anywhere):
#   bash run_baseline.sh smoke     # 1 epoch, seed 0, both markets  (pipeline check, ~minutes)
#   bash run_baseline.sh full      # 5 seeds x 40 epochs, both markets  (LONG, GPU-intensive)
#
# Migrated from qlib-MASTER's MASTER baseline wrapper.  It deliberately uses
# the rvqlab environment and exposes only AlphaMaster/src on PYTHONPATH.
# Runtime files are written below AlphaMaster/artifacts/.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-smoke}"

run_market () {
    local market=$1
    if [ "$MODE" = "smoke" ]; then
        echo "=== [SMOKE] $market (1 epoch, seed 0) ==="
        conda run -n rvqlab --no-capture-output python -u -m alphamaster.run_baseline --market "$market" --smoke
    elif [ "$MODE" = "full" ]; then
        # Run each seed in a FRESH python process. Running multiple seeds in one process
        # accumulates RAM (dataset/model/recorder not fully released between seeds) and can
        # OOM on large markets — e.g. SP500 peaks ~11GB/seed and exhausts a 15GB machine by
        # seed 2. One process per seed fully releases memory between seeds.
        echo "=== [FULL] $market (5 seeds x 40 epochs, one fresh process per seed) ==="
        for s in 0 1 2 3 4; do
            echo "--- $market seed $s ---"
            conda run -n rvqlab --no-capture-output python -u -m alphamaster.run_baseline --market "$market" --seed-start "$s" --seeds 1
        done
    else
        echo "Unknown mode: '$MODE' (use 'smoke' or 'full')"; exit 1
    fi
}

run_market csi300
run_market sp500
echo "=== done ($MODE) ==="
