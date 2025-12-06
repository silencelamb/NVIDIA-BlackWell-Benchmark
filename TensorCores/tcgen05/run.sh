#!/bin/bash
set -euo pipefail

# Build and run tcgen05 benchmarks.
# Usage:
#   ./run.sh ldst [iters] [shape]
#   ./run.sh cp_shift [iters]
#   ./run.sh mma_ws [iters] [kind]   # kind: f16|bf16|tf32|...
# shape: 0(.16x64b) 1(.16x128b) 2(.16x256b) 3(.32x32b) 4(.16x32bx2)

script_dir="$(cd "$(dirname "$0")" && pwd)"
mode="${1:-}"
log_dir="${script_dir}/log"
mkdir -p "${log_dir}"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/tcgen05_${mode:-unknown}_${timestamp}.log"

{
case "${mode}" in
  ldst)
    make -C "${script_dir}" ldst -j
    iters=${2:-1024}
    shape=${3:-0}
    echo "Running tcgen05 ldst iters=${iters} shape=${shape}"
    "${script_dir}/tmem_ldst" "$iters" "$shape"
    ;;
  cp_shift)
    make -C "${script_dir}" cp_shift -j
    iters=${2:-1024}
    echo "Running tcgen05 cp_shift iters=${iters}"
    "${script_dir}/tmem_cp_shift" "$iters"
    ;;
  mma_ws)
    kind=${3:-f16}
    iters=${2:-1024}
    echo "Building tcgen05 mma_ws kind=${kind}"
    make -C "${script_dir}" mma_ws -j MMA_KIND="$kind"
    echo "Running tcgen05 mma_ws iters=${iters} kind=${kind}"
    "${script_dir}/mma_ws_tmem" "$iters"
    ;;
  *)
    echo "Usage: $0 {ldst|cp_shift|mma_ws} [...]"
    exit 1
    ;;
esac
} | tee "${log_file}"

echo "Logs saved to ${log_file}"
