#!/bin/bash
set -euo pipefail

# Wrapper to run CoWoS latency/bandwidth tests.
# Usage: ./run_all.sh [compute_dev] [mem_dev]
# Default: single device (compute_dev == mem_dev == 0). If two GPUs with P2P, set mem_dev to the other GPU.

compute_dev=${1:-0}
mem_dev=${2:-$compute_dev}

script_dir="$(cd "$(dirname "$0")" && pwd)"
log_dir="${script_dir}/log"
mkdir -p "${log_dir}"
log_file="${log_dir}/cowos_$(date +%Y%m%d_%H%M%S).log"

{
  echo "Building CoWoS remote benchmarks..."
  make -C "${script_dir}" -j

  echo "Running remote latency (compute=${compute_dev}, mem=${mem_dev})"
  "${script_dir}/remote_latency" "${compute_dev}" "${mem_dev}"

  echo "Running remote bandwidth (compute=${compute_dev}, mem=${mem_dev})"
  "${script_dir}/remote_bw" "${compute_dev}" "${mem_dev}"

  echo "Running HSI memmap latency (compute=${compute_dev}, mem=${mem_dev})"
  "${script_dir}/hsi_latency" "${compute_dev}" "${mem_dev}" 4096 $((1<<20)) memmap cv
} | tee "${log_file}"

echo "Logs saved to ${log_file}"
