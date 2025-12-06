#!/bin/bash
set -euo pipefail

# Build and run deviceQuery, saving output to log.
# Usage: ./run.sh

script_dir="$(cd "$(dirname "$0")" && pwd)"
log_dir="${script_dir}/log"
mkdir -p "${log_dir}"
log_file="${log_dir}/deviceQuery_$(date +%Y%m%d_%H%M%S).log"

{
  echo "Building deviceQuery..."
  make -C "${script_dir}" -j
  echo "Running deviceQuery..."
  "${script_dir}/deviceQuery"
} | tee "${log_file}"

echo "Logs saved to ${log_file}"
