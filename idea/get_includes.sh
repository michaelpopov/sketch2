#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_ROOT="$SCRIPT_DIR/../llama-cpp"
DEST="$SCRIPT_DIR/llama-include"

if [ ! -d "$LLAMA_ROOT/include" ]; then
    echo "Error: llama-cpp repo not found at $LLAMA_ROOT" >&2
    exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"

cp "$LLAMA_ROOT"/include/llama.h         "$DEST/"
cp "$LLAMA_ROOT"/ggml/include/ggml.h     "$DEST/"
cp "$LLAMA_ROOT"/ggml/include/ggml-alloc.h   "$DEST/"
cp "$LLAMA_ROOT"/ggml/include/ggml-backend.h "$DEST/"
cp "$LLAMA_ROOT"/ggml/include/ggml-cpu.h     "$DEST/"
cp "$LLAMA_ROOT"/ggml/include/ggml-opt.h     "$DEST/"
cp "$LLAMA_ROOT"/ggml/include/gguf.h         "$DEST/"

echo "Copied headers to $DEST:"
ls -la "$DEST/"
