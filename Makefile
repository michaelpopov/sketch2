# --- Configuration ---
BUILD_DIR := build-dbg

# --- Targets ---

# Default target (runs when you type 'make')
.PHONY: all
all: build

# Compiles the project using the specified build directory
.PHONY: build
build:
	cmake --build $(BUILD_DIR) --parallel $(nproc)

# Runs the test suite with failure output enabled
.PHONY: test
test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# Optimization: Cleaning the build directory
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
