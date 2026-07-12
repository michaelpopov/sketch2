# --- Configuration ---
TYPE ?= rel
CMAKE_GENERATOR ?= Ninja

BUILD_DIR_dbg := build-dbg
BUILD_DIR_rel := build

BIN_DIR_dbg := bin-dbg
BIN_DIR_rel := bin

CMAKE_BUILD_TYPE_dbg := Debug
CMAKE_BUILD_TYPE_rel := Release

INSTALL_DIR := install
INSTALL_INCLUDE_DIR := $(INSTALL_DIR)/include
INSTALL_LIB_DIR := $(INSTALL_DIR)/lib
INSTALL_BIN_DIR := $(INSTALL_DIR)/bin

BUILD_DIR := $(BUILD_DIR_$(TYPE))
BIN_DIR := $(BIN_DIR_$(TYPE))
CMAKE_BUILD_TYPE := $(CMAKE_BUILD_TYPE_$(TYPE))

ALL_BUILD_DIRS := \
	$(BUILD_DIR_dbg) \
	$(BUILD_DIR_rel)
ALL_BIN_DIRS := \
	$(BIN_DIR_dbg) \
	$(BIN_DIR_rel)

ifeq ($(BUILD_DIR),)
$(error Unsupported TYPE='$(TYPE)'. Use TYPE={dbg,rel})
endif

JOBS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

# Optional per-arch tuning pass-throughs. Left empty by default so portable
# builds (Apple Silicon under Parallels, generic x86_64 servers) keep working
# unchanged.
ARM_SVE ?=
ARM_MCPU ?=
X86_MARCH ?=
X86_MTUNE ?=
SKETCH_EXTRA_FLAGS :=
ifneq ($(strip $(ARM_SVE)),)
SKETCH_EXTRA_FLAGS += -DSKETCH_ENABLE_ARM_SVE=ON
endif
ifneq ($(strip $(ARM_MCPU)),)
SKETCH_EXTRA_FLAGS += -DSKETCH_ARM_MCPU=$(ARM_MCPU)
endif
ifneq ($(strip $(X86_MARCH)),)
SKETCH_EXTRA_FLAGS += -DSKETCH_X86_MARCH=$(X86_MARCH)
endif
ifneq ($(strip $(X86_MTUNE)),)
SKETCH_EXTRA_FLAGS += -DSKETCH_X86_MTUNE=$(X86_MTUNE)
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
		'  make build TYPE={dbg,rel}' \
		'  make test TYPE={dbg,rel}' \
		'  make install' \
		'' \
		'Variables:' \
		'  TYPE     Build type selector. Default: rel' \
		'  CMAKE_GENERATOR CMake generator for repo build dirs. Default: Ninja' \
		'  JOBS     Parallelism for cmake --build. Default: host CPU count' \
		'  ARM_SVE   Set to 1 on AArch64 to enable SVE kernels (Graviton3+/AmpereOne)' \
		'  ARM_MCPU  Optional -mcpu tuning value (e.g. neoverse-v1, neoverse-n1, native)' \
		'  X86_MARCH Optional -march value on x86_64 (e.g. native, znver4, icelake-server)' \
		'  X86_MTUNE Optional -mtune value on x86_64 (scheduler tuning only; portable binary)' \
		'' \
		'Main targets:' \
		'  help          Show this summary' \
		'  build         Build the selected TYPE runtime' \
		'  build-arm-sve Release build with ARM SVE kernels; combine with ARM_MCPU=<tune>' \
		'  test          Build and run ctest for the selected TYPE runtime' \
		'  fetch-test-models Download the pinned embedding test model (enables E2E embed tests)' \
		'  install       Install release headers and runtime into install' \
		'  pytest     Run Python tests against the release runtime' \
			'  pydemo     Run the Python demo' \
			'  tut        Run all tutorials against the release runtime' \
			'  validate   Run the default validation flow' \
			'  clean      Remove contents of all build/bin directories' \
		'' \
		'Compatibility aliases:' \
		'  rel rtest'

# Compiles the project for the selected TYPE.
# Always re-run configuration so the directory matches the requested build type.
# Example: make build TYPE=dbg
.PHONY: build rel build-arm-sve
build:
	cmake -S . -B $(BUILD_DIR) -G "$(CMAKE_GENERATOR)" -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) $(SKETCH_EXTRA_FLAGS)
	@test -d $(BIN_DIR) || mkdir -p $(BIN_DIR)
	cmake --build $(BUILD_DIR) --parallel $(JOBS)
	cp $(BUILD_DIR)/lib/libsketch2.so $(BIN_DIR)/libsketch2.so

rel:
	$(MAKE) build TYPE=rel

# Release build with SVE kernels enabled for Graviton3/3E/4 and AmpereOne.
# Pair with ARM_MCPU=<tune> for Neoverse-specific scheduling, e.g.:
#   make build-arm-sve ARM_MCPU=neoverse-v1
build-arm-sve:
	$(MAKE) build TYPE=rel ARM_SVE=1 ARM_MCPU=$(ARM_MCPU)

# Runs the test suite for the selected TYPE.
# Example: make test TYPE=dbg
.PHONY: test rtest
test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

rtest:
	$(MAKE) test TYPE=rel

# Downloads the pinned embedding model fixture so `make test` covers the E2E
# embedding tests. CI runs this and sets SKETCH2_REQUIRE_EMBED_TESTS=1.
.PHONY: fetch-test-models
fetch-test-models:
	scripts/fetch-embed-test-model.sh $(BUILD_DIR)/test-models/all-MiniLM-L6-v2-Q4_K_M.gguf

# Installs the release runtime directory.
# The staged tree contains the public C header, the shared library, and the
# Python-facing helper script.
# Example: make install
.PHONY: install
install:
	$(MAKE) test TYPE=rel
	@mkdir -p $(INSTALL_INCLUDE_DIR) $(INSTALL_LIB_DIR) $(INSTALL_BIN_DIR)
	@find "$(INSTALL_INCLUDE_DIR)" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	@find "$(INSTALL_LIB_DIR)" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	@find "$(INSTALL_BIN_DIR)" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	cp src/sketch2api/sketch2.h $(INSTALL_INCLUDE_DIR)/
	cp $(BUILD_DIR_rel)/lib/libsketch2.so $(INSTALL_LIB_DIR)/
	cp src/pytest/sketch2_wrapper.py $(INSTALL_BIN_DIR)/

# Runs Python API tests against the release runtime
.PHONY: pytest
pytest: rel
	SKETCH2_LIB="$(BIN_DIR_rel)" python3 -m unittest discover -s src/pytest -p 'test_*.py'

# Runs Python demo that bulk-loads vectors and validates KNN output
.PHONY: pydemo
pydemo: rel
	SKETCH2_LIB="$(BIN_DIR_rel)" python3 src/pytest/demo.py

# Runs all tutorial scripts end-to-end
.PHONY: tut
tut: rel
	SKETCH2_LIB="$(BIN_DIR_rel)" python3 tutorial/run_all.py

# Runs the default validation flow, including the C++ suite plus the Python and
# tutorial flows that depend on the release runtime.
.PHONY: validate
validate: test pytest tut

# Removes the contents of build and runtime output directories for all build types.
.PHONY: clean
clean:
		for dir in \
			$(foreach dir,$(ALL_BUILD_DIRS) $(ALL_BIN_DIRS),"$(dir)" ) ; do \
			if [ -d "$$dir" ]; then \
				find "$$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +; \
			fi; \
		done
