#!/bin/bash
set -euo pipefail

# Build and run tcgen05 benchmarks.
# Usage:
#   ./run.sh ldst [iters] [shape]
#   ./run.sh cp_shift [iters]
#   ./run.sh mma_ws [iters]
# shape: 0(.16x64b) 1(.16x128b) 2(.16x256b) 3(.32x32b) 4(.16x32bx2)

case "${1:-}" in
  ldst)
    make -C "$(dirname "$0")" ldst -j
    iters=${2:-1024}
    shape=${3:-0}
    ./tmem_ldst "$iters" "$shape"
    ;;
  cp_shift)
    make -C "$(dirname "$0")" cp_shift -j
    iters=${2:-1024}
    ./tmem_cp_shift "$iters"
    ;;
  mma_ws)
    make -C "$(dirname "$0")" mma_ws -j
    iters=${2:-1024}
    ./mma_ws_tmem "$iters"
    ;;
  *)
    echo "Usage: $0 {ldst|cp_shift|mma_ws} [...]"
    exit 1
    ;;
esac
