#!/bin/bash
# ============================================================
#  run_pattern_finding_themachine.sh
#  Runs the pattern finding pipeline directly on themachine
#  (king.cs.ohio.edu NFS home — no Slurm needed).
#
#  Usage (from themachine):
#    bash /home/mg546924/llm_beatmap_generator/run_pattern_finding_themachine.sh
# ============================================================

set -euo pipefail

REPO_DIR="/home/mg546924/llm_beatmap_generator"
SCRIPT="$REPO_DIR/pattern_finding_approach/pattern_finding.py"
TARGET_DIR="$REPO_DIR/src/musicForBeatmap"
LOG_DIR="$REPO_DIR/logs"

# ── Python: prefer deepresonance_env, fall back to system python3 ──────────
PYTHON_CANDIDATES=(
    "/data/mg546924/conda_envs/deepresonance_env/bin/python"
    "/home/mg546924/.conda/envs/deepresonance_env/bin/python"
    "/home/mg546924/miniconda3/envs/deepresonance_env/bin/python"
    "$(which python3 2>/dev/null || echo '')"
)
PYTHON=""
for p in "${PYTHON_CANDIDATES[@]}"; do
    if [ -n "$p" ] && [ -x "$p" ]; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Could not find a valid Python interpreter. Aborting."
    exit 1
fi

echo "=============================================="
echo "  Pattern Finding Pipeline — themachine"
echo "  Repo:   $REPO_DIR"
echo "  Target: $TARGET_DIR"
echo "  Python: $PYTHON"
echo "  Start:  $(date)"
echo "=============================================="

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pattern_finding_$(date +%Y%m%d_%H%M%S).log"

cd "$REPO_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_DIR:$REPO_DIR/src"

echo "Logging to: $LOG_FILE"
echo ""

PYTHONUNBUFFERED=1 "$PYTHON" -u \
    "$SCRIPT" \
    --target_dir "$TARGET_DIR" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "  Finished: $(date)"
echo "  Log saved: $LOG_FILE"
echo "=============================================="
