#!/usr/bin/env bash
# Quick Phase 4.5 status check — run this when user wakes up.
# Shows: which capture is running, what's complete, what results are ready.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo "Phase 4.5 status — $(date)"
echo "=========================================="

echo ""
echo "[1] Trace files:"
ls -la traces/ntruplus768/phase45/ 2>&1 | grep -v "^total"

echo ""
echo "[2] Result files:"
ls -la results/ntruplus/phase45/ 2>&1 | grep -v "^total"

echo ""
echo "[3] Running processes:"
echo "  - scout/main capture:"
ps aux | grep -E "n18_phase4_g78_capture" | grep -v grep | awk '{print "    PID="$2" %CPU="$3" %MEM="$4" elapsed="$10" cmd="$11" "$12" "$13" "$14}'
echo "  - chain wrapper:"
ps aux | grep -E "run_phase45_chain" | grep -v grep | awk '{print "    PID="$2" elapsed="$10}'
echo "  - summary wrapper:"
ps aux | grep -E "n49_phase45_summary|phase45_summary_wrapper" | grep -v grep | awk '{print "    PID="$2" elapsed="$10}'

echo ""
echo "[4] Latest chain log:"
echo "    /tmp/phase45_chain.log:"
tail -10 /tmp/phase45_chain.log 2>/dev/null | sed 's/^/    /'

echo ""
echo "[5] Latest scout capture log:"
tail -5 /tmp/scout_capture.log 2>/dev/null | sed 's/^/    /' || echo "    (empty - python -u was not used)"

echo ""
echo "[6] Phase 4.5 summary file (if exists):"
if [ -f results/ntruplus/phase45/SUMMARY.md ]; then
    head -30 results/ntruplus/phase45/SUMMARY.md | sed 's/^/    /'
else
    echo "    (not yet generated)"
fi

echo ""
echo "[7] Phase 4.5 figures:"
ls results/ntruplus/phase45/figures/ 2>&1 | sed 's/^/    /'

echo ""
echo "=========================================="
echo "Quick actions:"
echo "  rerun analysis:  python3 scripts/n45_singleVictim_combine.py traces/ntruplus768/phase45/*.npz"
echo "  regen figures:   python3 scripts/n49_phase45_summary.py"
echo "  view summary:    cat results/ntruplus/phase45/SUMMARY.md"
echo "=========================================="
