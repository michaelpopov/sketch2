# --- Configuration ---
BUILD_DBG := build-dbg
BUILD_REL := build
BUILD_SAN := build-san
BUILD_DBG_NK := build-nk-dbg
BUILD_REL_NK := build-nk
BUILD_SAN_NK := build-nk-san
JOBS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
GBENCH_ESSENTIAL_MIN_TIME ?= 0.005s
GBENCH_EXTENDED_MIN_TIME ?= 0.05s
BENCH_TMPDIR ?= /tmp

# --- Targets ---

# Default target (runs when you type 'make')
.PHONY: all
all: build

# --- Build directory initialization ---
# Always re-run configuration so the directory matches the requested build type.

# Prepare required dependencies on Ubuntu
.PHONY: prepare
prepare:
	sudo apt update && sudo apt install -y build-essential cmake ninja-build -y

.PHONY: initdbg
initdbg:
	cmake -S . -B $(BUILD_DBG) -DCMAKE_BUILD_TYPE=Debug

.PHONY: initdbg-nk
initdbg-nk:
	cmake -S . -B $(BUILD_DBG_NK) -DCMAKE_BUILD_TYPE=Debug -DSKETCH_COMPUTE_ENGINE=numkong

.PHONY: initrel
initrel:
	cmake -S . -B $(BUILD_REL) -DCMAKE_BUILD_TYPE=Release

.PHONY: initrel-nk
initrel-nk:
	cmake -S . -B $(BUILD_REL_NK) -DCMAKE_BUILD_TYPE=Release -DSKETCH_COMPUTE_ENGINE=numkong

.PHONY: initsan
initsan:
	cmake -S . -B $(BUILD_SAN) -DCMAKE_BUILD_TYPE=Sanitizer

.PHONY: initsan-nk
initsan-nk:
	cmake -S . -B $(BUILD_SAN_NK) -DCMAKE_BUILD_TYPE=Sanitizer -DSKETCH_COMPUTE_ENGINE=numkong

# Compiles the project in debug build (initializes build-dbg if needed)
.PHONY: build
build: initdbg
	@test -d bin-dbg || mkdir -p bin-dbg
	@test -d "$(BUILD_DBG)" || mkdir -p "$(BUILD_DBG)"
	cmake --build $(BUILD_DBG) --parallel $(JOBS)

.PHONY: build-nk
build-nk: initdbg-nk
	@test -d bin-dbg-nk || mkdir -p bin-dbg-nk
	@test -d "$(BUILD_DBG_NK)" || mkdir -p "$(BUILD_DBG_NK)"
	cmake --build $(BUILD_DBG_NK) --parallel $(JOBS)

# Compiles the project in release build (initializes build if needed)
.PHONY: rel
rel: initrel
	@test -d bin || mkdir -p bin
	@test -d "$(BUILD_REL)" || mkdir -p "$(BUILD_REL)"
	cmake --build $(BUILD_REL) --parallel $(JOBS)

.PHONY: rel-nk
rel-nk: initrel-nk
	@test -d bin-nk || mkdir -p bin-nk
	@test -d "$(BUILD_REL_NK)" || mkdir -p "$(BUILD_REL_NK)"
	cmake --build $(BUILD_REL_NK) --parallel $(JOBS)

# Compiles the project in sanitizer build (initializes build-san if needed)
.PHONY: san
san: initsan
	@test -d "$(BUILD_SAN)" || mkdir -p "$(BUILD_SAN)"
	cmake --build $(BUILD_SAN) --parallel $(JOBS)

.PHONY: san-nk
san-nk: initsan-nk
	@test -d bin-san-nk || mkdir -p bin-san-nk
	@test -d "$(BUILD_SAN_NK)" || mkdir -p "$(BUILD_SAN_NK)"
	cmake --build $(BUILD_SAN_NK) --parallel $(JOBS)

# Runs the test suite with failure output enabled
.PHONY: test
test: build
	ctest --test-dir $(BUILD_DBG) --output-on-failure

.PHONY: test-nk
test-nk: build-nk
	ctest --test-dir $(BUILD_DBG_NK) --output-on-failure

# Runs the standalone thread-pool unit tests on demand
.PHONY: tpooltest
tpooltest: build
	bin-dbg/utest_thread_pool

# Runs the sketch2api unit test binary on demand
.PHONY: sketch2test
sketch2test: build
	bin-dbg/utest_sketch2

# Runs the test suite in release build
.PHONY: rtest
rtest: rel
	ctest --test-dir $(BUILD_REL) --output-on-failure

.PHONY: rtest-nk
rtest-nk: rel-nk
	ctest --test-dir $(BUILD_REL_NK) --output-on-failure

# Installs the public header and release shared library under install/
.PHONY: install
install: rtest
	@mkdir -p install/include install/lib
	cp src/sketch2api/sketch2.h install/include/
	cp $(BUILD_REL)/lib/libsketch2.so install/lib/

# Runs the test suite in sanitizer build
.PHONY: santest
santest: san
	ctest --test-dir $(BUILD_SAN) --output-on-failure

.PHONY: santest-nk
santest-nk: san-nk
	ctest --test-dir $(BUILD_SAN_NK) --output-on-failure

# Runs Python API tests
.PHONY: pytest
pytest:
	python3 -m unittest discover -s src/pytest -p 'test_*.py'

# Runs Python demo that bulk-loads vectors and validates KNN output
.PHONY: pydemo
pydemo:
	python3 src/pytest/demo.py

# Runs all tutorial scripts end-to-end
.PHONY: tut
tut: build
	python3 tutorial/run_all.py

# Runs the Python demo against the release libsketch2 artifact
.PHONY: demo
demo: rel
	SKETCH2_LOG_LEVEL=DEBUG \
	SKETCH2_THREAD_POOL_SIZE=12 \
	python3 src/pytest/demo.py \
		--dim 256 \
		--k 10 \
		--range-size 1M \
		--binary \
		--dist-func L2 \
		--sketch2-lib $(BUILD_REL)/lib/libsketch2.so \
		--extension-lib $(BUILD_REL)/lib/libsketch2.so

# Configures the release benchmark build with Google Benchmark enabled.
.PHONY: benchcfg
benchcfg:
	cmake -S . -B $(BUILD_REL) -DCMAKE_BUILD_TYPE=Release -DSKETCH_ENABLE_BENCHMARKS=ON

# Builds the release benchmark binaries.
.PHONY: benchbuild
benchbuild: benchcfg
	cmake --build $(BUILD_REL) --parallel $(JOBS) --target bench_compute

# Compatibility alias for the remaining calc benchmark workflow.
.PHONY: bench
bench: benchrel

.PHONY: benchrel
benchrel: benchbuild
	@echo "gbench_comp was removed; use bin/bench_compute with explicit arguments"

# Compatibility alias for the remaining calc benchmark workflow.
.PHONY: benchext
benchext: benchbuild
	@echo "gbench_comp was removed; use bin/bench_compute with explicit arguments"

# Compatibility alias for the remaining calc benchmark workflow.
.PHONY: benchcomp
benchcomp: benchbuild
	@echo "bench_comp was removed; use bin/bench_compute with explicit arguments"

# Compatibility alias for the remaining calc benchmark workflow.
.PHONY: ds_bench
ds_bench: benchbuild
	@echo "gbench_comp was removed; use bin/bench_compute with explicit arguments"

# Compatibility alias for the remaining calc benchmark workflow.
.PHONY: ds_mix_bench
ds_mix_bench: benchbuild
	@echo "gbench_comp was removed; use bin/bench_compute with explicit arguments"

# Runs full local coverage flow:
# - debug unit tests
# - release unit tests
# - Python integration tests
# - Python demo
# - dataset benchmark slices
.PHONY: cover
cover:
	$(MAKE) test
	$(MAKE) rtest
	$(MAKE) santest
	$(MAKE) test-nk
	$(MAKE) rtest-nk
	$(MAKE) santest-nk
	$(MAKE) pytest
	$(MAKE) demo
	$(MAKE) ds_bench
	$(MAKE) ds_mix_bench
	$(MAKE) tut
	rm -r /tmp/sketch2_test_data_*
	rm -r /tmp/sketch2_tutorial

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

# Removes debug and release build artifacts so the next make/make rel rebuilds
# from scratch and repopulates bin-dbg/ and bin/ with fresh binaries.
.PHONY: clear
clear:
	rm -rf "$(BUILD_DBG)" "$(BUILD_REL)" bin-dbg bin
