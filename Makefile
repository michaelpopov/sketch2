# --- Configuration ---
TYPE ?= rel
ENGINE ?= hwy
CMAKE_GENERATOR ?= Ninja

BUILD_DIR_dbg_hwy := build-dbg
BUILD_DIR_rel_hwy := build
BUILD_DIR_dbg_nk := build-nk-dbg
BUILD_DIR_rel_nk := build-nk

BIN_DIR_dbg_hwy := bin-dbg-hwy
BIN_DIR_rel_hwy := bin-hwy
BIN_DIR_dbg_nk := bin-dbg-nk
BIN_DIR_rel_nk := bin-nk

CMAKE_BUILD_TYPE_dbg := Debug
CMAKE_BUILD_TYPE_rel := Release

CMAKE_ENGINE_FLAG_hwy := -DSKETCH2_COMPUTE_ENGINE=highway
CMAKE_ENGINE_FLAG_nk := -DSKETCH2_COMPUTE_ENGINE=numkong

INSTALL_DIR_hwy := install-hwy
INSTALL_DIR_nk := install-nk

BUILD_DIR := $(BUILD_DIR_$(TYPE)_$(ENGINE))
BIN_DIR := $(BIN_DIR_$(TYPE)_$(ENGINE))
CMAKE_BUILD_TYPE := $(CMAKE_BUILD_TYPE_$(TYPE))
CMAKE_ENGINE_FLAG := $(CMAKE_ENGINE_FLAG_$(ENGINE))
INSTALL_DIR := $(INSTALL_DIR_$(ENGINE))

ALL_BUILD_DIRS := \
	$(BUILD_DIR_dbg_hwy) \
	$(BUILD_DIR_rel_hwy) \
	$(BUILD_DIR_dbg_nk) \
	$(BUILD_DIR_rel_nk)
ALL_BIN_DIRS := \
	$(BIN_DIR_dbg_hwy) \
	$(BIN_DIR_rel_hwy) \
	$(BIN_DIR_dbg_nk) \
	$(BIN_DIR_rel_nk)

ifeq ($(BUILD_DIR),)
$(error Unsupported TYPE='$(TYPE)' ENGINE='$(ENGINE)'. Use TYPE={dbg,rel} and ENGINE={hwy,nk})
endif

JOBS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

# Optional AArch64 tuning pass-throughs. Left empty by default so portable builds
# (including Apple Silicon under Parallels) keep working unchanged.
ARM_SVE ?=
ARM_MCPU ?=
SKETCH_EXTRA_FLAGS :=
ifneq ($(strip $(ARM_SVE)),)
SKETCH_EXTRA_FLAGS += -DSKETCH_ENABLE_ARM_SVE=ON
endif
ifneq ($(strip $(ARM_MCPU)),)
SKETCH_EXTRA_FLAGS += -DSKETCH_ARM_MCPU=$(ARM_MCPU)
endif

# --- Targets ---

# Default target (runs when you type 'make')
.PHONY: all help
all: build

help:
	@printf '%s\n' \
		'Sketch2 Make targets' \
		'' \
		'Preferred parameterized usage:' \
		'  make build' \
		'  make test' \
		'  make build TYPE={dbg,rel} ENGINE={hwy,nk}' \
		'  make test TYPE={dbg,rel} ENGINE={hwy,nk}' \
		'  make install ENGINE={hwy,nk}' \
		'' \
		'Variables:' \
		'  TYPE     Build type selector. Default: rel' \
		'  ENGINE   Compute engine selector. Default: hwy' \
		'  CMAKE_GENERATOR CMake generator for repo build dirs. Default: Ninja' \
		'  JOBS     Parallelism for cmake --build. Default: host CPU count' \
		'  ARM_SVE  Set to 1 on AArch64 to enable SVE kernels (Graviton3+/AmpereOne)' \
		'  ARM_MCPU Optional -mcpu tuning value (e.g. neoverse-v1, neoverse-n1, native)' \
		'' \
		'Main targets:' \
		'  help          Show this summary' \
		'  build         Build the selected TYPE/ENGINE runtime' \
		'  build-arm-sve Release build with ARM SVE kernels; combine with ARM_MCPU=<tune>' \
		'  test          Build and run ctest for the selected TYPE/ENGINE' \
		'  install       Install release headers and runtime for ENGINE into install-{hwy,nk}' \
		'  pytest     Run Python tests against the highway release runtime' \
			'  pydemo     Run the Python demo' \
			'  tut        Run all tutorials against the highway release runtime' \
			'  hwy        Run the highway validation flow' \
			'  nk         Run the numkong validation flow' \
			'  clean      Remove contents of all build/bin directories' \
		'' \
		'Compatibility aliases:' \
		'  build-nk rel rel-nk' \
		'  test-nk rtest rtest-nk' \
		'  install-hwy install-nk'

# Compiles the project for the selected TYPE/ENGINE pair.
# Always re-run configuration so the directory matches the requested build type.
# Example: make build ENGINE=nk
.PHONY: build build-nk rel rel-nk build-arm-sve
build:
	cmake -S . -B $(BUILD_DIR) -G "$(CMAKE_GENERATOR)" -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) $(CMAKE_ENGINE_FLAG) $(SKETCH_EXTRA_FLAGS)
	@test -d $(BIN_DIR) || mkdir -p $(BIN_DIR)
	cmake --build $(BUILD_DIR) --parallel $(JOBS)
	cp $(BUILD_DIR)/lib/libsketch2.so $(BIN_DIR)/libsketch2.so

build-nk:
	$(MAKE) build ENGINE=nk

rel:
	$(MAKE) build TYPE=rel ENGINE=hwy

rel-nk:
	$(MAKE) build TYPE=rel ENGINE=nk

# Release build with SVE kernels enabled for Graviton3/3E/4 and AmpereOne.
# Pair with ARM_MCPU=<tune> for Neoverse-specific scheduling, e.g.:
#   make build-arm-sve ARM_MCPU=neoverse-v1
build-arm-sve:
	$(MAKE) build TYPE=rel ENGINE=$(ENGINE) ARM_SVE=1 ARM_MCPU=$(ARM_MCPU)

# Runs the test suite for the selected TYPE/ENGINE pair.
# Example: make test ENGINE=nk
.PHONY: test test-nk rtest rtest-nk
test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-nk:
	$(MAKE) test ENGINE=nk

rtest:
	$(MAKE) test TYPE=rel ENGINE=hwy

rtest-nk:
	$(MAKE) test TYPE=rel ENGINE=nk

# Installs the release runtime directory for the selected engine.
# The staged tree contains the public C header plus the Python-facing runtime.
# Example: make install ENGINE=nk
.PHONY: install install-hwy install-nk
install:
	$(MAKE) test TYPE=rel ENGINE=$(ENGINE)
	@mkdir -p $(INSTALL_DIR)/include $(INSTALL_DIR)/bin
	@find "$(INSTALL_DIR)/include" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	@find "$(INSTALL_DIR)/bin" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	cp src/sketch2api/sketch2.h $(INSTALL_DIR)/include/
	cp $(BIN_DIR_rel_$(ENGINE))/libsketch2.so $(INSTALL_DIR)/bin/
	cp src/pytest/sketch2_wrapper.py $(INSTALL_DIR)/bin/

install-hwy:
	$(MAKE) install ENGINE=hwy

install-nk:
	$(MAKE) install ENGINE=nk

# Runs Python API tests against the highway release runtime
.PHONY: pytest
pytest: rel
	SKETCH2_LIB="$(BIN_DIR_rel_hwy)" python3 -m unittest discover -s src/pytest -p 'test_*.py'

# Runs Python demo that bulk-loads vectors and validates KNN output
.PHONY: pydemo
pydemo: rel
	SKETCH2_LIB="$(BIN_DIR_rel_hwy)" python3 src/pytest/demo.py

# Runs all tutorial scripts end-to-end
.PHONY: tut
tut: rel
	SKETCH2_LIB="$(BIN_DIR_rel_hwy)" python3 tutorial/run_all.py

# Runs the highway portion of the full validation flow.
# Includes the default highway C++ suite plus the Python and tutorial flows
# that depend on the highway release runtime.
.PHONY: hwy
hwy: test pytest tut

# Runs the NumKong portion of the full validation flow.
# Includes the default NumKong C++ suite.
.PHONY: nk
nk: test-nk

# Removes the contents of build and runtime output directories for all build types and engines.
.PHONY: clean
clean:
		for dir in \
			$(foreach dir,$(ALL_BUILD_DIRS) $(ALL_BIN_DIRS),"$(dir)" ) ; do \
			if [ -d "$$dir" ]; then \
				find "$$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +; \
			fi; \
		done
