#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${BUILD_DIR:-$root/build}"
configuration="${CONFIGURATION:-Release}"
mlir_dir="${MLIR_DIR:-$root/llvm-project/install/lib/cmake/mlir}"
python_args=()
if [[ -x "$root/.venv/bin/python" ]]; then
    python_args=(-DPython_EXECUTABLE="$root/.venv/bin/python")
fi

if [[ ! -f "$mlir_dir/MLIRConfig.cmake" ]]; then
    echo "MLIRConfig.cmake not found under '$mlir_dir'. Set MLIR_DIR explicitly." >&2
    exit 1
fi

cmake -S "$root" -B "$build_dir" \
    -DMLIR_DIR="$mlir_dir" \
    -DCMAKE_BUILD_TYPE="$configuration" \
    "${python_args[@]}"
cmake --build "$build_dir" --config "$configuration" --parallel

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    ctest --test-dir "$build_dir" -C "$configuration" --output-on-failure
fi
