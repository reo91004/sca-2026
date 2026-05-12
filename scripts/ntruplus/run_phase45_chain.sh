#!/usr/bin/env bash
# Phase 4.5 chain: wait scout → analyze → start main capture.
# Designed to run in background while user is away.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCOUT="traces/ntruplus768/phase45/scout_K1_L16_N8.npz"
MAIN="traces/ntruplus768/phase45/main_K1_L24_N16.npz"
SCOUT_OUT="results/ntruplus/phase45/scout_K1_L16_N8"
MAIN_OUT="results/ntruplus/phase45/main_K1_L24_N16"
COMBINED_OUT="results/ntruplus/phase45/combined"
LOG_DIR="results/ntruplus/phase45/logs"

mkdir -p "$(dirname "$SCOUT_OUT")" "$LOG_DIR"

# Wait for scout
echo "[$(date)] waiting for scout: $SCOUT"
while [ ! -f "$SCOUT" ]; do sleep 60; done
echo "[$(date)] scout completed."

# Scout analysis (M1+M5+full stack)
echo "[$(date)] running scout analysis (n43)"
python3 -u scripts/n43_singleVictim_multilane.py \
    --input "$SCOUT" --out-prefix "$SCOUT_OUT" \
    > "$LOG_DIR/scout_eval.log" 2>&1
ec=$?
echo "[$(date)] scout n43 exit=$ec"

# Combine scout (single-victim, 1 batch)
python3 -u scripts/n45_singleVictim_combine.py "$SCOUT" \
    --out-prefix "$COMBINED_OUT" \
    > "$LOG_DIR/combine_scout.log" 2>&1
echo "[$(date)] combined (scout only) → $COMBINED_OUT.{npz,md}"

# Decide whether to start main: M1 top-100 ≥ 1
M1_T100=$(python3 -c "
import numpy as np
rows = np.load('$SCOUT_OUT.npz', allow_pickle=True)['rows']
t100 = sum(1 for r in rows if r['rk_M1_pred'] < 100)
print(t100)
" 2>/dev/null)

echo "[$(date)] scout M1 baseline top-100 = $M1_T100"

if [ -z "$M1_T100" ] || [ "$M1_T100" -eq 0 ]; then
    echo "[$(date)] WARN: scout 0 top-100 — skipping main capture, idle"
    echo "[$(date)] DONE (scout-only)" >> "$LOG_DIR/chain.log"
    exit 0
fi

# Start main capture
echo "[$(date)] starting main capture (K=1 L=24 N=16) ~5h ETA"
LANES_MAIN="0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184"
python3 -u scripts/n18_phase4_g78_capture.py \
    -K 1 -L "$LANES_MAIN" -N 16 -o "$MAIN" \
    > "$LOG_DIR/main_capture.log" 2>&1
ec=$?
echo "[$(date)] main capture exit=$ec"

# Main analysis + final combine
if [ -f "$MAIN" ]; then
    python3 -u scripts/n43_singleVictim_multilane.py \
        --input "$MAIN" --out-prefix "$MAIN_OUT" \
        > "$LOG_DIR/main_eval.log" 2>&1
    python3 -u scripts/n45_singleVictim_combine.py "$SCOUT" "$MAIN" \
        --out-prefix "$COMBINED_OUT" \
        > "$LOG_DIR/combine_final.log" 2>&1
    echo "[$(date)] main analysis + combined → $COMBINED_OUT"
fi

echo "[$(date)] CHAIN DONE" >> "$LOG_DIR/chain.log"
