#!/bin/bash
set -euo pipefail

# Wrapper to run CoWoS latency/bandwidth tests.
# Usage: ./run_all.sh [compute_dev] [mem_dev]
# Default: single device (compute_dev == mem_dev == 0). If two GPUs with P2P, set mem_dev to the other GPU.

compute_dev=${1:-0}
mem_dev=${2:-$compute_dev}

echo "Building CoWoS remote benchmarks..."
make -C "$(dirname "$0")" -j

echo "Running remote latency (compute=${compute_dev}, mem=${mem_dev})"
./remote_latency "${compute_dev}" "${mem_dev}"

echo "Running remote bandwidth (compute=${compute_dev}, mem=${mem_dev})"
./remote_bw "${compute_dev}" "${mem_dev}"
