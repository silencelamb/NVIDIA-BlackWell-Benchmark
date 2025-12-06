#!/bin/bash
set -euo pipefail

# Build all regular unit benchmarks and save logs.
# Usage:
#   ./run_all.sh            # build and run all binaries in ./bin
#   ./run_all.sh build      # build only

script_dir="$(cd "$(dirname "$0")" && pwd)"
log_dir="${script_dir}/log"
mkdir -p "${log_dir}"
log_file="${log_dir}/regular_$(date +%Y%m%d_%H%M%S).log"
action="${1:-run}"

{
  echo "Building RegularUnits..."
  make -C "${script_dir}" -j

  if [[ "${action}" == "run" ]]; then
    echo "Running binaries under ${script_dir}/bin (errors do not stop the log)..."
    shopt -s nullglob
    for bin in "${script_dir}"/bin/*; do
      if [[ -x "${bin}" ]]; then
        echo "==> $(basename "${bin}")"
        "${bin}" || echo "  (failed: $?)"
      fi
    done
    shopt -u nullglob
  fi
} | tee "${log_file}"

echo "Logs saved to ${log_file}"
