#!/bin/bash
set -euo pipefail

# Simple wrapper to run remote latency and bandwidth tests.
# Usage: ./run_all.sh [compute_dev] [mem_dev]

compute_dev=${1:-0}
mem_dev=${2:-1}

echo "Building CoWoS remote benchmarks..."
make -C "$(dirname "$0")" -j

echo "Running remote latency (compute=${compute_dev}, mem=${mem_dev})"
./remote_latency "${compute_dev}" "${mem_dev}"

echo "Running remote bandwidth (compute=${compute_dev}, mem=${mem_dev})"
./remote_bw "${compute_dev}" "${mem_dev}"
