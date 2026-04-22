# --- Configuration ---
TYPE ?= rel
ENGINE ?= hwy

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
		'  TYPE   Build type selector. Default: rel' \
		'  ENGINE Compute engine selector. Default: hwy' \
		'  JOBS   Parallelism for cmake --build. Default: host CPU count' \
		'' \
		'Main targets:' \
		'  help       Show this summary' \
		'  build      Build the selected TYPE/ENGINE runtime' \
		'  test       Build and run ctest for the selected TYPE/ENGINE' \
		'  install    Install release headers and runtime for ENGINE into install-{hwy,nk}' \
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
.PHONY: build build-nk rel rel-nk
build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) $(CMAKE_ENGINE_FLAG)
	@test -d $(BIN_DIR) || mkdir -p $(BIN_DIR)
	cmake --build $(BUILD_DIR) --parallel $(JOBS)
	cp $(BUILD_DIR)/lib/libsketch2.so $(BIN_DIR)/libsketch2.so

build-nk:
	$(MAKE) build ENGINE=nk

rel:
	$(MAKE) build TYPE=rel ENGINE=hwy

rel-nk:
	$(MAKE) build TYPE=rel ENGINE=nk

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
