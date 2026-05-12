#!/usr/bin/env bash
# Phase 4.5 main capture — K=1 fixed sk, 24 lanes (stride 8 incl 0/64/80/128),
# G=78 (HW1+HW2), N=16. ETA ~5h.
# Use python -u for unbuffered stdout so we can monitor progress.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# stride 8: 0, 8, 16, ..., 184 → 24 lanes (incl 0, 64, 80, 128 calibrated)
LANES="0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184"
OUT="traces/ntruplus768/phase45/main_K1_L24_N16.npz"

mkdir -p "$(dirname "$OUT")"

echo "[INFO] Main capture: K=1 L=24 N=16 G=78 → ${OUT}"
echo "[INFO] Lanes: $LANES"

exec python3 -u scripts/ntruplus/n18_phase4_g78_capture.py \
    -K 1 \
    -L "$LANES" \
    -N 16 \
    -o "$OUT"
