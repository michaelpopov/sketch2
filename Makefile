# --- Configuration ---
BUILD_DBG := build-dbg
BUILD_REL := build
BUILD_SAN := build-san
JOBS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
GBENCH_ESSENTIAL_MIN_TIME ?= 0.005s
GBENCH_EXTENDED_MIN_TIME ?= 0.05s
BENCH_TMPDIR ?= /tmp

# --- Targets ---

# Default target (runs when you type 'make')
.PHONY: all
all: build

# --- Build directory initialization ---
# These are real-file targets keyed on CMakeCache.txt so cmake only runs once.
# Use 'make initdbg / initrel / initsan' to re-run configuration explicitly.

$(BUILD_DBG)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DBG) -DCMAKE_BUILD_TYPE=Debug

$(BUILD_REL)/CMakeCache.txt:
	cmake -S . -B $(BUILD_REL) -DCMAKE_BUILD_TYPE=Release

$(BUILD_SAN)/CMakeCache.txt:
	cmake -S . -B $(BUILD_SAN) -DCMAKE_BUILD_TYPE=Sanitizer

.PHONY: initdbg
initdbg: $(BUILD_DBG)/CMakeCache.txt

.PHONY: initrel
initrel: $(BUILD_REL)/CMakeCache.txt

.PHONY: initsan
initsan: $(BUILD_SAN)/CMakeCache.txt

# Compiles the project in debug build (initializes build-dbg if needed)
.PHONY: build
build: $(BUILD_DBG)/CMakeCache.txt
	cmake --build $(BUILD_DBG) --parallel $(JOBS)

# Compiles the project in release build (initializes build if needed)
.PHONY: rel
rel: $(BUILD_REL)/CMakeCache.txt
	cmake --build $(BUILD_REL) --parallel $(JOBS)

# Compiles the project in sanitizer build (initializes build-san if needed)
.PHONY: san
san: $(BUILD_SAN)/CMakeCache.txt
	cmake --build $(BUILD_SAN) --parallel $(JOBS)

# Runs the test suite with failure output enabled
.PHONY: test
test: build
	ctest --test-dir $(BUILD_DBG) --output-on-failure

# Runs the standalone thread-pool unit tests on demand
.PHONY: tpooltest
tpooltest:
	bin-dbg/utest_thread_pool

# Runs the test suite in release build
.PHONY: rtest
rtest: rel
	ctest --test-dir $(BUILD_REL) --output-on-failure

# Runs the test suite in sanitizer build
.PHONY: santest
santest: san
	ctest --test-dir $(BUILD_SAN) --output-on-failure

# Runs Python API tests
.PHONY: pytest
pytest:
	python3 -m unittest discover -s src/pytest -p 'test_*.py'

# Runs Python demo that bulk-loads vectors and validates KNN output
.PHONY: pydemo
pydemo:
	python3 src/pytest/demo.py

# Runs the Python demo against the release libparasol/libvlite artifacts
.PHONY: demo
demo: rel
	SKETCH2_LOG_LEVEL=DEBUG \
	SKETCH2_THREAD_POOL_SIZE=12 \
	python3 src/pytest/demo.py \
		--count 10M \
		--dim 256 \
		--k 10 \
		--range-size 100K \
		--binary \
		--dist-func L2 \
		--parasol-lib $(BUILD_REL)/lib/libparasol.so \
		--vlite-lib $(BUILD_REL)/lib/libvlite.so

# Configures the release benchmark build with Google Benchmark enabled.
.PHONY: benchcfg
benchcfg:
	cmake -S . -B $(BUILD_REL) -DCMAKE_BUILD_TYPE=Release -DSKETCH_ENABLE_BENCHMARKS=ON

# Builds the release benchmark binaries.
.PHONY: benchbuild
benchbuild: benchcfg
	cmake --build $(BUILD_REL) --parallel $(JOBS) --target bench_comp gbench_comp

# Runs the Google Benchmark-based compute/scanner benchmark suite in release mode.
.PHONY: bench
bench: benchrel

.PHONY: benchrel
benchrel: benchbuild
	TMPDIR=$(BENCH_TMPDIR) SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_min_time=$(GBENCH_ESSENTIAL_MIN_TIME)

# Runs the extended Google Benchmark suite in release mode.
.PHONY: benchext
benchext: benchbuild
	TMPDIR=$(BENCH_TMPDIR) SKETCH2_GBENCH_PROFILE=extended bin/gbench_comp --benchmark_min_time=$(GBENCH_EXTENDED_MIN_TIME)

# Runs the lightweight compute benchmark in release mode.
.PHONY: benchcomp
benchcomp: benchbuild
	bin/bench_comp

# Runs the essential Google Benchmark suite restricted to the reader scanner mode.
.PHONY: reader_bench
reader_bench: benchbuild
	TMPDIR=$(BENCH_TMPDIR) SKETCH2_GBENCH_PROFILE=essential SKETCH2_GBENCH_SCANNER_MODE=reader bin/gbench_comp --benchmark_min_time=$(GBENCH_ESSENTIAL_MIN_TIME)

# Runs the essential Google Benchmark suite restricted to the dataset_persisted scanner mode.
.PHONY: ds_bench
ds_bench: benchbuild
	TMPDIR=$(BENCH_TMPDIR) SKETCH2_GBENCH_PROFILE=essential SKETCH2_GBENCH_SCANNER_MODE=dataset_persisted bin/gbench_comp --benchmark_min_time=$(GBENCH_ESSENTIAL_MIN_TIME)

# Runs the essential Google Benchmark suite restricted to the dataset_mixed scanner mode.
.PHONY: ds_mix_bench
ds_mix_bench: benchbuild
	TMPDIR=$(BENCH_TMPDIR) SKETCH2_GBENCH_PROFILE=essential SKETCH2_GBENCH_SCANNER_MODE=dataset_mixed bin/gbench_comp --benchmark_min_time=$(GBENCH_ESSENTIAL_MIN_TIME)

# Runs Python shell with Sketch2 objects ready
.PHONY: pyshell
pyshell:
	python3 src/pytest/shell.py --db-root /tmp/sketch2_db
# python3 src/pytest/shell.py --db-root /tmp/skdb --dataset demo --create
# python3 src/pytest/shell.py --db-root /tmp/skdb --dataset demo

# Optimization: Cleaning the build directory
.PHONY: clean
clean:
	@if [ -d "$(BUILD_DBG)" ]; then \
		find "$(BUILD_DBG)" -type f \( -name '*.o' -o -name '*.obj' \) -delete; \
	fi
