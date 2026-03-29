#!/usr/bin/env bash
# Runs GPU PTQ design experiment, captures logs, and filters for config failure info.

set -euo pipefail

# Always run from the repo root (directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/data/output/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Patterns to extract from logs:
#   - Skipping lines (config failures with reasons)
#   - CUDA errors
#   - Worker errors/exits
#   - cutlass / TMA / ONNX errors
#   - Wrote/No results collected (outcome confirmation)
# ---------------------------------------------------------------------------
FILTER_PATTERN="Skipping|CUDA error|Worker|cutlass|TMA|INVALID_GRAPH|cudaError\
|IllegalInstruction|RuntimeError|AssertionError|Traceback|Error:|Exception\
|Wrote|No .* results collected|Aborting|Too many"

run_experiment() {
    local script_name="$1"
    local label="$2"

    local raw_log="${LOG_DIR}/${label}_raw_${TIMESTAMP}.txt"
    local filtered_log="${LOG_DIR}/${label}_filtered_${TIMESTAMP}.txt"

    echo "========================================"
    echo "Starting: ${label}"
    echo "Script:   scripts/${script_name}"
    echo "Raw log:  ${raw_log}"
    echo "Filtered: ${filtered_log}"
    echo "Working dir: $(pwd)"
    echo "========================================"

    # Run script, tee stdout+stderr to raw log, also show in terminal
    if python -m "scripts.${script_name%.py}" > >(tee "${raw_log}") 2>&1; then
        echo "[${label}] Completed successfully."
    else
        echo "[${label}] Exited with non-zero status — check raw log for details."
    fi

    # Filter raw log for config failure/outcome signals
    echo "--- Filtered log: ${label} (${TIMESTAMP}) ---" > "${filtered_log}"
    echo "--- Pattern: ${FILTER_PATTERN} ---"           >> "${filtered_log}"
    echo ""                                              >> "${filtered_log}"

    grep -E "${FILTER_PATTERN}" "${raw_log}" >> "${filtered_log}" || \
        echo "(No matching lines found)" >> "${filtered_log}"

    echo "[${label}] Filtered log written to: ${filtered_log}"
    echo ""
}

# ---------------------------------------------------------------------------
# Run GPU design experiment
# ---------------------------------------------------------------------------
run_experiment "design_space_sampling_experiment.py" "GPU Config Design Experiment"
echo "Logs written to: ${LOG_DIR}"