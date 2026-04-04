#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SIMPLE_SMOKE_TEST_DB_DIR="$(mktemp -d /tmp/sketch2_simple_smoke.XXXXXX)"
export SIMPLE_SMOKE_TEST_DATASET="simple_smoke"
export SIMPLE_SMOKE_TEST_DIMS="256"
export SIMPLE_SMOKE_TEST_COUNT="90000"
export SIMPLE_SMOKE_TEST_SLEEP="1"
export SIMPLE_SMOKE_TEST_READERS="8"
export SIMPLE_SMOKE_TEST_K="20"
export SIMPLE_SMOKE_TEST_TYPE="f16"
export SIMPLE_SMOKE_TEST_DIST="l2"
export SIMPLE_SMOKE_TEST_RANGE_SIZE="10000"
export SIMPLE_SMOKE_TEST_LOG_LEVEL="ERROR"
export SIMPLE_SMOKE_TEST_THREAD_POOL_SIZE="16"
export SKETCH2_CONFIG="${SIMPLE_SMOKE_TEST_DB_DIR}/config.ini"
readonly LOG_DIR="${SIMPLE_SMOKE_TEST_DB_DIR}/logs"

export SIMPLE_SMOKE_TEST_REPEAT="4"

pids=()

cleanup() {
    local pid
    # First ask children to terminate.
    for pid in "${pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    # Then wait so we don't leave orphaned writers/readers behind.
    for pid in "${pids[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
}

trap cleanup EXIT INT TERM

echo "[driver] simple smoke test configuration"
echo "[driver]   db_dir=${SIMPLE_SMOKE_TEST_DB_DIR}"
echo "[driver]   dataset=${SIMPLE_SMOKE_TEST_DATASET}"
echo "[driver]   dims=${SIMPLE_SMOKE_TEST_DIMS}"
echo "[driver]   count=${SIMPLE_SMOKE_TEST_COUNT}"
echo "[driver]   sleep=${SIMPLE_SMOKE_TEST_SLEEP}"
echo "[driver]   repeat=${SIMPLE_SMOKE_TEST_REPEAT}"
echo "[driver]   readers=${SIMPLE_SMOKE_TEST_READERS}"
echo "[driver]   k=${SIMPLE_SMOKE_TEST_K}"

cd "${SCRIPT_DIR}"
python3 initializer.py

mkdir -p "${LOG_DIR}"
echo "[driver]   log_dir=${LOG_DIR}"

python3 writer.py >"${LOG_DIR}/writer.stdout.log" 2>"${LOG_DIR}/writer.stderr.log" &
pids+=("$!")
echo "[driver] started writer pid=${pids[-1]} stdout=${LOG_DIR}/writer.stdout.log stderr=${LOG_DIR}/writer.stderr.log"

for ((reader_index = 1; reader_index <= SIMPLE_SMOKE_TEST_READERS; ++reader_index)); do
    python3 reader.py --reader-id "reader-${reader_index}" \
        >"${LOG_DIR}/reader-${reader_index}.stdout.log" \
        2>"${LOG_DIR}/reader-${reader_index}.stderr.log" &
    pids+=("$!")
    echo "[driver] started reader-${reader_index} pid=${pids[-1]} stdout=${LOG_DIR}/reader-${reader_index}.stdout.log stderr=${LOG_DIR}/reader-${reader_index}.stderr.log"
done

status=0
for pid in "${pids[@]}"; do
    if wait "${pid}"; then
        :
    else
        status=$?
        echo "[driver] process pid=${pid} failed"
        cleanup
        break
    fi
done

if [[ "${status}" -eq 0 ]]; then
    echo "[driver] all smoke-test processes completed successfully"
else
    exit "${status}"
fi
