#!/usr/bin/env bash
# Fetch the pinned, SHA-256-verified embedding-model fixture used by the
# utest_embed end-to-end tests.
#
# Usage: fetch-embed-test-model.sh [output-path]
#   output-path defaults to <repo>/build/test-models/${MODEL_FILE}
set -euo pipefail

MODEL_REPO="second-state/All-MiniLM-L6-v2-Embedding-GGUF"
MODEL_REVISION="544f204f2eaa2d71361ffc74d6df7170285b286a"
MODEL_FILE="all-MiniLM-L6-v2-Q4_K_M.gguf"
MODEL_SHA256="2ec4cee28a27a9c973d5f5230930d6ef6e52694bd2bc71be26a9bef5b1d755e6"
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-${script_dir}/../build/test-models/${MODEL_FILE}}"

verify() {
    echo "${MODEL_SHA256}  $1" | sha256sum --check --status
}

if [[ -f "${out}" ]] && verify "${out}"; then
    echo "embed test model already present: ${out}"
    exit 0
fi

mkdir -p "$(dirname "${out}")"
tmp="${out}.download.$$"
trap 'rm -f "${tmp}"' EXIT

echo "fetching ${MODEL_URL}"
curl --location --fail --silent --show-error --retry 3 -o "${tmp}" "${MODEL_URL}"

if ! verify "${tmp}"; then
    echo "error: checksum mismatch for ${MODEL_FILE} (expected ${MODEL_SHA256})" >&2
    exit 1
fi

mv "${tmp}" "${out}"
trap - EXIT
echo "embed test model ready: ${out}"
