#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Internal defaults for this run. We export so child processes see them.
export SKETCH2_CONFIG_ROOT="${SKETCH2_CONFIG_ROOT:-$(mktemp -d /tmp/sketch2_COMPUTE_PERF.XXXXXX)}"
export COMPUTE_PERF_SKIP_INIT="${COMPUTE_PERF_SKIP_INIT:-0}"
export SKETCH2_CONFIG="${SKETCH2_CONFIG_ROOT}/config.ini"
export COMPUTE_PERF_TEST_DATASET="${COMPUTE_PERF_TEST_DATASET:-perf_test}"
export COMPUTE_PERF_TEST_DIMS="${COMPUTE_PERF_TEST_DIMS:-256}"
export COMPUTE_PERF_TEST_COUNT="${COMPUTE_PERF_TEST_COUNT:-100000}"
export COMPUTE_PERF_TEST_REPEAT="${COMPUTE_PERF_TEST_REPEAT:-10}"
export COMPUTE_PERF_TEST_K="${COMPUTE_PERF_TEST_K:-20}"
export COMPUTE_PERF_TEST_TYPE="${COMPUTE_PERF_TEST_TYPE:-f32}"
export COMPUTE_PERF_TEST_DIST="${COMPUTE_PERF_TEST_DIST:-cos,l2,dot}"
export COMPUTE_PERF_TEST_RANGE_SIZE="${COMPUTE_PERF_TEST_RANGE_SIZE:-10000}"
export COMPUTE_PERF_TEST_LOG_LEVEL="${COMPUTE_PERF_TEST_LOG_LEVEL:-ERROR}"
export COMPUTE_PERF_TEST_THREAD_POOL_SIZE="${COMPUTE_PERF_TEST_THREAD_POOL_SIZE:-1}"
export COMPUTE_PERF_TEST_ENGINES="${COMPUTE_PERF_TEST_ENGINES:-scalar,auto,highway,numkong}"
export COMPUTE_PERF_TEST_BENCHMARKS="${COMPUTE_PERF_TEST_BENCHMARKS:-scan,kernel}"
export COMPUTE_PERF_KERNEL_ITERATIONS="${COMPUTE_PERF_KERNEL_ITERATIONS:-200000}"
export COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS="${COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS:-5000}"
export COMPUTE_PERF_KERNEL_REPEATS="${COMPUTE_PERF_KERNEL_REPEATS:-7}"
export COMPUTE_PERF_TEST_CLEANUP="${COMPUTE_PERF_TEST_CLEANUP:-0}"
export DUMMY_CALC="${DUMMY_CALC:-0}"

cleanup() {
    if [[ "${COMPUTE_PERF_TEST_CLEANUP}" == "1" ]]; then
        echo "[driver] cleaning up ${SKETCH2_CONFIG_ROOT}..."
        rm -rf "${SKETCH2_CONFIG_ROOT}"
    else
        echo "[driver] preserving ${SKETCH2_CONFIG_ROOT} (set COMPUTE_PERF_TEST_CLEANUP=1 to remove it automatically)"
    fi
}

trap cleanup EXIT INT TERM

readonly LOG_DIR="${SKETCH2_CONFIG_ROOT}/logs"
readonly DIAG_DIR="${LOG_DIR}/diag"
mkdir -p "${LOG_DIR}"
mkdir -p "${DIAG_DIR}"
export COMPUTE_PERF_DIAG_DIR="${DIAG_DIR}"

# Ask the shell to keep core dumps when the platform permits it.
ulimit -c unlimited 2>/dev/null || true

echo "[driver] performance test configuration"
echo "[driver]   config_root=${SKETCH2_CONFIG_ROOT}"
echo "[driver]   skip_init=${COMPUTE_PERF_SKIP_INIT}"
echo "[driver]   dataset=${COMPUTE_PERF_TEST_DATASET}"
echo "[driver]   dims=${COMPUTE_PERF_TEST_DIMS}"
echo "[driver]   count=${COMPUTE_PERF_TEST_COUNT}"
echo "[driver]   repeat=${COMPUTE_PERF_TEST_REPEAT}"
echo "[driver]   k=${COMPUTE_PERF_TEST_K}"
echo "[driver]   type=${COMPUTE_PERF_TEST_TYPE}"
echo "[driver]   dist=${COMPUTE_PERF_TEST_DIST}"
echo "[driver]   range_size=${COMPUTE_PERF_TEST_RANGE_SIZE}"
echo "[driver]   engines=${COMPUTE_PERF_TEST_ENGINES}"
echo "[driver]   benchmarks=${COMPUTE_PERF_TEST_BENCHMARKS}"
echo "[driver]   kernel_iterations=${COMPUTE_PERF_KERNEL_ITERATIONS}"
echo "[driver]   kernel_warmup_iterations=${COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS}"
echo "[driver]   kernel_repeats=${COMPUTE_PERF_KERNEL_REPEATS}"
echo "[driver]   cleanup=${COMPUTE_PERF_TEST_CLEANUP}"
echo "[driver]   dummy_calc=${DUMMY_CALC}"
echo "[driver]   diag_dir=${COMPUTE_PERF_DIAG_DIR}"
echo "[driver]   core_limit=$(ulimit -c)"
if [[ -r /proc/sys/kernel/core_pattern ]]; then
    echo "[driver]   core_pattern=$(cat /proc/sys/kernel/core_pattern)"
fi

cd "${SCRIPT_DIR}"

write_run_env() {
    {
        echo "SKETCH2_CONFIG_ROOT=${SKETCH2_CONFIG_ROOT}"
        echo "SKETCH2_CONFIG=${SKETCH2_CONFIG}"
        echo "COMPUTE_PERF_SKIP_INIT=${COMPUTE_PERF_SKIP_INIT}"
        echo "COMPUTE_PERF_TEST_DATASET=${COMPUTE_PERF_TEST_DATASET}"
        echo "COMPUTE_PERF_TEST_DIMS=${COMPUTE_PERF_TEST_DIMS}"
        echo "COMPUTE_PERF_TEST_COUNT=${COMPUTE_PERF_TEST_COUNT}"
        echo "COMPUTE_PERF_TEST_REPEAT=${COMPUTE_PERF_TEST_REPEAT}"
        echo "COMPUTE_PERF_TEST_K=${COMPUTE_PERF_TEST_K}"
        echo "COMPUTE_PERF_TEST_TYPE=${COMPUTE_PERF_TEST_TYPE}"
        echo "COMPUTE_PERF_TEST_DIST=${COMPUTE_PERF_TEST_DIST}"
        echo "COMPUTE_PERF_TEST_RANGE_SIZE=${COMPUTE_PERF_TEST_RANGE_SIZE}"
        echo "COMPUTE_PERF_TEST_LOG_LEVEL=${COMPUTE_PERF_TEST_LOG_LEVEL}"
        echo "COMPUTE_PERF_TEST_THREAD_POOL_SIZE=${COMPUTE_PERF_TEST_THREAD_POOL_SIZE}"
        echo "COMPUTE_PERF_TEST_ENGINES=${COMPUTE_PERF_TEST_ENGINES}"
        echo "COMPUTE_PERF_TEST_BENCHMARKS=${COMPUTE_PERF_TEST_BENCHMARKS}"
        echo "COMPUTE_PERF_KERNEL_ITERATIONS=${COMPUTE_PERF_KERNEL_ITERATIONS}"
        echo "COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS=${COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS}"
        echo "COMPUTE_PERF_KERNEL_REPEATS=${COMPUTE_PERF_KERNEL_REPEATS}"
        echo "COMPUTE_PERF_TEST_CLEANUP=${COMPUTE_PERF_TEST_CLEANUP}"
        echo "DUMMY_CALC=${DUMMY_CALC}"
        echo "COMPUTE_PERF_DIAG_DIR=${COMPUTE_PERF_DIAG_DIR}"
    } > "${LOG_DIR}/run_env.txt"
}

write_run_env

IFS=',' read -ra DIST_FUNCS <<< "${COMPUTE_PERF_TEST_DIST}"
IFS=',' read -ra ENGINES <<< "${COMPUTE_PERF_TEST_ENGINES}"
dist_index=0
for dist in "${DIST_FUNCS[@]}"; do
    dist=$(echo "${dist}" | xargs)
    echo "[driver] preparing dataset for dist=${dist}"

    if [[ "${COMPUTE_PERF_SKIP_INIT}" == "1" ]]; then
        echo "[driver] skipping initializer.py for dist=${dist} as requested..."
        if [[ ! -d "${SKETCH2_CONFIG_ROOT}" ]]; then
            echo "[driver] ERROR: SKETCH2_CONFIG_ROOT does not exist: ${SKETCH2_CONFIG_ROOT}"
            exit 1
        fi
    else
        if [[ ${dist_index} -eq 0 ]]; then
            export COMPUTE_PERF_INIT_PRESERVE_ROOT="0"
        else
            export COMPUTE_PERF_INIT_PRESERVE_ROOT="1"
        fi
        export COMPUTE_PERF_SINGLE_DIST="${dist}"
        set +e
        python3 initializer.py 2>&1 | tee "${LOG_DIR}/initializer_${dist}.log"
        init_rc=${PIPESTATUS[0]}
        set -e
        if [[ ${init_rc} -ne 0 ]]; then
            echo "[driver] ERROR: initializer.py failed for dist=${dist} with exit code ${init_rc}. See ${LOG_DIR}/initializer_${dist}.log"
            exit 1
        fi
    fi

    # initializer may recreate the config root from scratch, so ensure the log dir exists again.
    mkdir -p "${LOG_DIR}"
    mkdir -p "${DIAG_DIR}"
    write_run_env

    dataset_path="${SKETCH2_CONFIG_ROOT}/${COMPUTE_PERF_TEST_DATASET}_${dist}"
    if [[ ! -d "${dataset_path}" ]]; then
        echo "[driver] ERROR: expected dataset directory not found: ${dataset_path}"
        exit 1
    fi
    gt_path="${SKETCH2_CONFIG_ROOT}/ground_truth_${dist}.json"
    if [[ ! -f "${gt_path}" ]]; then
        echo "[driver] ERROR: expected ground truth file not found: ${gt_path}"
        exit 1
    fi
    echo "[driver] initializer finished successfully and verified dataset for dist=${dist}"

    for engine in "${ENGINES[@]}"; do
        engine=$(echo "${engine}" | xargs) # trim whitespace
        echo "[driver] benchmarks engine=${engine} dist=${dist}"

        if [[ "${engine}" == "auto" ]]; then
            unset SKETCH2_COMPUTE_ENGINE
        else
            export SKETCH2_COMPUTE_ENGINE="${engine}"
        fi
        export COMPUTE_PERF_SINGLE_DIST="${dist}"
        set +e
        python3 runner.py 2>&1 | tee "${LOG_DIR}/runner_${engine}_${dist}.log"
        runner_rc=${PIPESTATUS[0]}
        set -e
        if [[ ${runner_rc} -ne 0 ]]; then
            echo "[driver] ERROR: runner.py failed for engine=${engine} dist=${dist} with exit code ${runner_rc}. See ${LOG_DIR}/runner_${engine}_${dist}.log"
            echo "[driver] diagnostics directory: ${DIAG_DIR}"
            if ls "${DIAG_DIR}"/diag_"${engine}"_*.json >/dev/null 2>&1; then
                echo "[driver] diagnostic state files:"
                ls "${DIAG_DIR}"/diag_"${engine}"_*.json
            fi
            if ls "${DIAG_DIR}"/repro_"${engine}"_*.sh >/dev/null 2>&1; then
                echo "[driver] repro scripts:"
                ls "${DIAG_DIR}"/repro_"${engine}"_*.sh
            fi
            if ls "${DIAG_DIR}"/repro_loop_"${engine}"_*.sh >/dev/null 2>&1; then
                echo "[driver] repro loop scripts:"
                ls "${DIAG_DIR}"/repro_loop_"${engine}"_*.sh
            fi
            exit 1
        fi
    done

    if [[ "${COMPUTE_PERF_SKIP_INIT}" != "1" ]]; then
        echo "[driver] removing dataset artifacts for dist=${dist}"
        rm -rf "${dataset_path}"
        rm -f "${gt_path}"
        rm -f "${SKETCH2_CONFIG_ROOT}/input_${dist}.txt"
    fi

    dist_index=$((dist_index + 1))
done

unset COMPUTE_PERF_SINGLE_DIST
unset COMPUTE_PERF_INIT_PRESERVE_ROOT

echo "[driver] generating summary report..."
set +e
python3 reporter.py 2>&1 | tee "${LOG_DIR}/reporter.log"
reporter_rc=${PIPESTATUS[0]}
set -e
if [[ ${reporter_rc} -ne 0 ]]; then
    echo "[driver] ERROR: reporter.py failed with exit code ${reporter_rc}. See ${LOG_DIR}/reporter.log"
    exit 1
fi

echo "[driver] performance test completed"
echo "[driver] logs available in ${LOG_DIR}"
