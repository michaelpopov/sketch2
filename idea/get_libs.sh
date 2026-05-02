#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_BUILD="$SCRIPT_DIR/../llama-cpp/build/bin"
DEST="$SCRIPT_DIR/llama-libs"

if [ ! -d "$LLAMA_BUILD" ]; then
    echo "Error: llama-cpp build directory not found at $LLAMA_BUILD" >&2
    echo "Build llama-cpp first." >&2
    exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"

# Copy the real .so files under their bare names (no version suffixes).
# We load by exact name via dlopen, so soname versioning is unnecessary.
for lib in libllama.so libllama-common.so libggml.so libggml-base.so libggml-cpu.so; do
    # Follow the symlink chain to get the actual file.
    src=$(readlink -f "$LLAMA_BUILD/$lib")
    cp "$src" "$DEST/$lib"
done

echo "Copied libraries to $DEST:"
ls -la "$DEST/"
