# --- Configuration ---
BUILD_DIR := build-dbg
JOBS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

# --- Targets ---

# Default target (runs when you type 'make')
.PHONY: all
all: build

# Compiles the project using the specified build directory
.PHONY: build
build:
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

# Runs the test suite with failure output enabled
.PHONY: test
test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# Runs Python API tests
.PHONY: pytest
pytest:
	python3 -m unittest discover -s src/pytest -p 'test_*.py'

# Runs Python demo that bulk-loads vectors and validates KNN output
.PHONY: pydemo
pydemo:
	python3 src/pytest/demo.py

# Runs Python shell with Sketch2 objects ready
.PHONY: pyshell
pyshell:
	python3 src/pytest/shell.py --db-root /tmp/sketch2_db
# python3 src/pytest/shell.py --db-root /tmp/skdb --dataset demo --create
# python3 src/pytest/shell.py --db-root /tmp/skdb --dataset demo

# Optimization: Cleaning the build directory
.PHONY: clean
clean:
	@if [ -d "$(BUILD_DIR)" ]; then \
		find "$(BUILD_DIR)" -type f \( -name '*.o' -o -name '*.obj' \) -delete; \
	fi
