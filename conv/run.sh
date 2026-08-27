#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

NUMA_NODE="${NUMA_NODE:-1}"

gcc -O3 -mcpu=native -msve-vector-bits=512 -funroll-loops \
    -DCONV_OUTPUT_BLOCK=80 \
    bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp

numactl -N "${NUMA_NODE}" ./conv2d_test 4096 6144 39 39 1
numactl -N "${NUMA_NODE}" ./conv2d_test 6144 4096 41 41 1
numactl -N "${NUMA_NODE}" ./conv2d_test 4256 6390 55 55 1
numactl -N "${NUMA_NODE}" ./conv2d_test 6390 4256 81 81 1
