#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <capture_dir> [--warmup N] [--iters N] [--limit N] [--sorted 0|1]" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BENCH_BIN=${GEGE_UNIQUE_BENCH_BIN:-${REPO_DIR}/build/gege/gege_unique_replay_bench}
BACKENDS=${GEGE_UNIQUE_BENCH_BACKENDS:-"hash cuco"}

for backend in ${BACKENDS}; do
  echo "backend=${backend}"
  GEGE_UNIQUE_BACKEND="${backend}" "${BENCH_BIN}" "$@"
done
